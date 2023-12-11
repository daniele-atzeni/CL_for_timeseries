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
        mean_layer,
        n_features,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling

        self.n_features = n_features  ## my code here

        with self.name_scope():
            self.mlp = mx.gluon.nn.HybridSequential()
            dims = self.num_hidden_dimensions
            for layer_no, units in enumerate(dims):
                self.mlp.add(mx.gluon.nn.Dense(units=units, activation="relu"))
                if self.batch_normalization:
                    self.mlp.add(mx.gluon.nn.BatchNorm())
            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * self.n_features))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(o, (-1, prediction_length, self.n_features))
                )
            )
            self.scaler = MeanScaler() if mean_scaling else NOPScaler()

            self.mean_layer = mean_layer  ## my code here

    def get_distr_args(
        self, F, past_target: Tensor, past_feat_dynamic_real: Tensor
    ) -> Tensor:
        """
        past_target (batch, context_length, n_features)
        past_feat_dynamic_real (batch, context_length, n_features*2)    # contains mean and vars
        scale_target as past_target
        target_scale (batch)
        mlp_outputs (batch, pred_length, last_net_hidden_dim)
        distr_args tuple (3 * (batch, pred_length))
        scale (batch, 1)
        loc (batch, 1)
        pred_means (batch, pred_length)
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
        past_target = (past_target - means) / (vars + 1e-8).sqrt()  # type: ignore MXNet typing issue
        past_target = past_target.flatten()

        scaled_target, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),
        )
        mlp_outputs = self.mlp(scaled_target)

        pred_means = self.mean_layer(means.flatten())  # type: ignore MXNet typing issue
        pred_means = pred_means.reshape((-1, self.prediction_length, self.n_features))

        return mlp_outputs + pred_means


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        past_feat_dynamic_real: Tensor,
    ) -> Tensor:
        prediction = self.get_distr_args(F, past_target, past_feat_dynamic_real)
        # (batch_size, prediction_length, target_dim)
        loss = (prediction - future_target).abs().mean(axis=-1).mean(axis=-1)  # type: ignore MXNet typing issue

        return loss


class SimpleFeedForwardPredictionNetwork(SimpleFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Tensor,
    ) -> Tensor:
        out = self.get_distr_args(F, past_target, past_feat_dynamic_real)

        return out
