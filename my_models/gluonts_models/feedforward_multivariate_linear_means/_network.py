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
        n_features,  ## my code here
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.distr_output = distr_output

        self.n_features = n_features  ## my code here

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
        self, F, past_target: Tensor, past_feat_dynamic_real: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        past_target (batch, context_length, n_features)
        past_feat_dynamic_real (batch, context_length, n_features*2)    # contains mean and vars
        mlp_outputs (batch, pred_length, last_net_hidden_dim)
        """

        means = past_feat_dynamic_real.slice(
            begin=(None, None, 0 * self.n_features),
            end=(None, None, 1 * self.n_features),
        )
        vars = past_feat_dynamic_real.slice(
            begin=(None, None, 1 * self.n_features),
            end=(None, None, 2 * self.n_features),
        )

        # normalize past_target
        past_target = (past_target - means) / (F.sqrt(vars) + 1e-8)
        past_target = past_target.flatten()

        mlp_outputs = self.mlp(past_target)

        distr_args = self.distr_args_proj(mlp_outputs)
        # with multivariate gaussian:
        # mu (batch, pred_length, n_feat), L (batch, pred_length, n_feat, n_feat)

        scale = None
        loc = None
        # Setting this to None avoids the call to AffineTransformedDistribution
        # which causes problems in the multivariate case

        pred_means = self.mean_layer(F.flatten(means))
        pred_means = pred_means.reshape((-1, self.prediction_length, self.n_features))
        # we add mean layer preds to the means predicted by the output distribution
        # i.e. the 0th element of distr_args
        distr_args = tuple(
            [el if i != 0 else el + pred_means for i, el in enumerate(distr_args)]
        )

        return distr_args, loc, scale  # type:ignore not my code


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_feat_dynamic_real: Tensor,
    ) -> Tensor:
        distr_args, loc, scale = self.get_distr_args(
            F, past_target, past_feat_dynamic_real
        )
        distr = self.distr_output.distribution(distr_args, loc=loc, scale=scale)
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
        past_feat_dynamic_real: Tensor,
    ) -> Tensor:
        distr_args, loc, scale = self.get_distr_args(
            F, past_target, past_feat_dynamic_real
        )
        distr = self.distr_output.distribution(distr_args, loc=loc, scale=scale)

        # (num_samples, batch_size, prediction_length, n_features)
        samples = distr.sample(self.num_parallel_samples)

        # (batch_size, num_samples, prediction_length, n_features)
        return samples.swapaxes(0, 1)  # type:ignore not my code


class SimpleFeedForwardDistributionNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(self, num_parallel_samples: int = 100, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self, F, past_target: Tensor, past_feat_dynamic_real: Tensor
    ) -> Tensor:
        distr_args, loc, scale = self.get_distr_args(
            F, past_target, past_feat_dynamic_real
        )
        return distr_args, loc, scale  # type:ignore not my code
