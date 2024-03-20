# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Tuple

import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.util import weighted_average


class SimpleFeedForwardNetworkBase(mx.gluon.HybridBlock):
    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        num_hidden_dimensions: List[int],
        prediction_length: int,
        context_length: int,
        batch_normalization: bool,
        mean_scaling: bool,
        distr_output: DistributionOutput,
        mean_layer,  ## my code here
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.distr_output = distr_output

        with self.name_scope():
            self.distr_args_proj = self.distr_output.get_args_proj()
            self.mlp = mx.gluon.nn.HybridSequential()
            dims = self.num_hidden_dimensions
            for layer_no, units in enumerate(dims[:-1]):
                self.mlp.add(mx.gluon.nn.Dense(units=units, activation="relu"))
                if self.batch_normalization:
                    self.mlp.add(mx.gluon.nn.BatchNorm())
            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * dims[-1]))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(o, (-1, prediction_length, dims[-1]))
                )
            )
            self.scaler = MeanScaler() if mean_scaling else NOPScaler()

            self.mean_layer = mean_layer  ## my code here

    def get_distr_args(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Tensor,
        feat_static_real: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        past_target (batch, context_length)
        past_feat_dynamic_real (batch, context_length, 2)    # contains mean and vars
        feat_static_real (batch, n_gas_params * 1)
        mlp_outputs (batch, pred_length, last_net_hidden_dim)
        """
        means = past_feat_dynamic_real.slice(
            begin=(None, None, 0),
            end=(None, None, 1),
        )  # (batch, context_length, 1)
        vars = past_feat_dynamic_real.slice(
            begin=(None, None, 1),
            end=(None, None, 2),
        )  # (batch, context_length, 1)

        # normalize past_target
        past_target = (past_target - F.squeeze(means)) / (
            F.sqrt(F.squeeze(vars)) + 1e-8
        )

        scaled_target, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),
        )
        mlp_outputs = self.mlp(scaled_target)

        distr_args = self.distr_args_proj(mlp_outputs)

        scale = target_scale.expand_dims(axis=1)
        loc = F.zeros_like(scale)

        # compute mean layer prediction
        pred_means, pred_vars = self.mean_layer(
            past_target, means, vars, feat_static_real
        )

        # add mean layer predictions to the predicted mean parameter of the distribution
        # mean parameter in our case is the 0th element of distr_args
        new_means = pred_vars.sqrt() * distr_args[0] + pred_means
        new_vars = pred_vars.sqrt() * distr_args[1]
        distr_args = (new_means, new_vars, distr_args[2])

        return distr_args, loc, scale  # type:ignore I know its a tuple of Tensors


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_means_vars: Tensor,
        gas_params: Tensor,
    ) -> Tensor:
        distr_args, loc, scale = self.get_distr_args(
            F,
            past_target,
            past_means_vars,
            gas_params,
        )
        distr = self.distr_output.distribution(distr_args, loc=loc, scale=scale)
        # (batch_size, prediction_length, target_dim)
        loss = distr.loss(future_target)

        return weighted_average(F=F, x=loss, weights=F.ones_like(loss), axis=1)


class SimpleFeedForwardSamplingNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_means_vars: Tensor,
        gas_params: Tensor,
    ) -> Tensor:
        distr_args, loc, scale = self.get_distr_args(
            F,
            past_target,
            past_means_vars,
            gas_params,
        )
        distr = self.distr_output.distribution(distr_args, loc=loc, scale=scale)

        # (num_samples, batch_size, prediction_length)
        samples = distr.sample(self.num_parallel_samples)

        # (batch_size, num_samples, prediction_length)
        return samples.swapaxes(0, 1)  # type:ignore not my code


class SimpleFeedForwardDistributionNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_means_vars: Tensor,
        gas_params: Tensor,
    ) -> Tensor:
        distr_args, loc, scale = self.get_distr_args(
            F,
            past_target,
            past_means_vars,
            gas_params,
        )
        return distr_args, loc, scale  # type:ignore not my code
