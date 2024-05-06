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
    """
    Abstract base class to implement feed-forward networks for probabilistic
    time series prediction.

    This class does not implement hybrid_forward: this is delegated
    to the two subclasses SimpleFeedForwardTrainingNetwork and
    SimpleFeedForwardPredictionNetwork, that define respectively how to
    compute the loss and how to generate predictions.

    Parameters
    ----------
    num_hidden_dimensions
        Number of hidden nodes in each layer.
    prediction_length
        Number of time units to predict.
    context_length
        Number of time units that condition the predictions.
    batch_normalization
        Whether to use batch normalization.
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse.
    distr_output
        Distribution to fit.
    kwargs
    """

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
                    lambda F, o: F.reshape(
                        o, (-1, prediction_length, dims[-1])
                    )
                )
            )
            self.scaler = MeanScaler() if mean_scaling else NOPScaler()

    def get_distr_args(
        self, F, past_target: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Given past target values, applies the feed-forward network and maps the
        output to the parameter of probability distribution for future
        observations.

        Parameters
        ----------
        F
        past_target
            Tensor containing past target observations.
            Shape: (batch_size, context_length, target_dim).

        Returns
        -------
        Tensor
            The parameters of distribution.
        Tensor
            An array containing the location (shift) of the distribution.
        Tensor
            An array containing the scale of the distribution.
        """
        scaled_target, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),
        )
        mlp_outputs = self.mlp(scaled_target)
        distr_args = self.distr_args_proj(mlp_outputs)
        scale = None
        loc = None
        return distr_args, loc, scale


class SimpleFeedForwardTrainingNetwork(SimpleFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        # future_observed_values: Tensor,
    ) -> Tensor:
        """
        Computes a probability distribution for future data given the past, and
        returns the loss associated with the actual future observations.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Tensor with future observations.
            Shape: (batch_size, prediction_length, target_dim).
        future_observed_values
            Tensor indicating which values in the target are observed, and
            which ones are imputed instead.

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size, ).
        """
        distr_args, loc, scale = self.get_distr_args(F, past_target)
        distr = self.distr_output.distribution(
            distr_args, loc=loc, scale=scale
        )

        # (batch_size, prediction_length, target_dim)
        loss = distr.loss(future_target)

        weighted_loss = weighted_average(
            F=F, x=loss, weights=F.ones_like(loss), axis=1
        )

        # (batch_size, )
        return weighted_loss


class SimpleFeedForwardSamplingNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(self, F, past_target: Tensor) -> Tensor:
        """
        Computes a probability distribution for future data given the past, and
        draws samples from it.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).

        Returns
        -------
        Tensor
            Prediction sample. Shape: (batch_size, samples, prediction_length).
        """

        distr_args, loc, scale = self.get_distr_args(F, past_target)
        distr = self.distr_output.distribution(
            distr_args, loc=loc, scale=scale
        )

        # (num_samples, batch_size, prediction_length)
        samples = distr.sample(self.num_parallel_samples)

        # (batch_size, num_samples, prediction_length)
        return samples.swapaxes(0, 1)


class SimpleFeedForwardDistributionNetwork(SimpleFeedForwardNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(self, F, past_target: Tensor) -> Tensor:
        """
        Computes the parameters of distribution for future data given the past,
        and draws samples from it.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).

        Returns
        -------
        Tensor
            The parameters of distribution.
        Tensor
            An array containing the location (shift) of the distribution.
        Tensor
            An array containing the scale of the distribution.
        """
        distr_args, loc, scale = self.get_distr_args(F, past_target)
        return distr_args, loc, scale

from functools import partial
from typing import List, Optional

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import (
    DataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from gluonts.model.forecast_generator import DistributionForecastGenerator
from gluonts.mx.batchify import batchify
from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.model.predictor import RepresentableBlockPredictor
from gluonts.mx.trainer import Trainer
from gluonts.mx.util import get_hybrid_forward_input_names
from gluonts.transform import (
    AddObservedValuesIndicator,
    ExpectedNumInstanceSampler,
    InstanceSampler,
    InstanceSplitter,
    SelectFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
)
from gluonts.transform.feature import (
    DummyValueImputation,
    MissingValueImputation,
)

# from ._network import (
#     SimpleFeedForwardDistributionNetwork,
#     SimpleFeedForwardSamplingNetwork,
#     SimpleFeedForwardTrainingNetwork,
# )


class SimpleFeedForwardEstimator(GluonEstimator):


    # The validated() decorator makes sure that parameters are checked by
    # Pydantic and allows to serialize/print models. Note that all parameters
    # have defaults except for `prediction_length`. which is
    # recommended in GluonTS to allow to compare models easily.
    @validated()
    def __init__(
        self,
        prediction_length: int,
        sampling: bool = True,
        trainer: Trainer = Trainer(),
        num_hidden_dimensions: Optional[List[int]] = None,
        context_length: Optional[int] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        imputation_method: Optional[MissingValueImputation] = None,
        batch_normalization: bool = False,
        mean_scaling: bool = True,
        num_parallel_samples: int = 100,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Defines an estimator.

        All parameters should be serializable.
        """
        super().__init__(trainer=trainer, batch_size=batch_size)

        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert num_hidden_dimensions is None or (
            [d > 0 for d in num_hidden_dimensions]
        ), "Elements of `num_hidden_dimensions` should be > 0"
        assert (
            num_parallel_samples > 0
        ), "The value of `num_parallel_samples` should be > 0"

        self.num_hidden_dimensions = (
            num_hidden_dimensions
            if num_hidden_dimensions is not None
            else list([40, 40])
        )
        self.prediction_length = prediction_length
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.distr_output = distr_output
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.num_parallel_samples = num_parallel_samples
        self.sampling = sampling
        self.imputation_method = (
            imputation_method
            if imputation_method is not None
            else DummyValueImputation(self.distr_output.value_in_support)
        )
        self.train_sampler = (
            train_sampler
            if train_sampler is not None
            else ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=prediction_length
            )
        )
        self.validation_sampler = (
            validation_sampler
            if validation_sampler is not None
            else ValidationSplitSampler(min_future=prediction_length)
        )

    # Here we do only a simple operation to convert the input data to a form
    # that can be digested by our model by only splitting the target in two, a
    # conditioning part and a to-predict part, for each training example.
    # For a more complex transformation example, see the `gluonts.model.deepar`
    # transformation that includes time features, age feature, observed values
    # indicator, ...
    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ],
            allow_missing=True,
        ) 
        # + AddObservedValuesIndicator(
        #     target_field=FieldName.TARGET,
        #     output_field=FieldName.OBSERVED_VALUES,
        #     dtype=self.dtype,
        #     imputation_method=self.imputation_method,
        # )   

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[],
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            SimpleFeedForwardTrainingNetwork
        )
        instance_splitter = self._create_instance_splitter("training")
        return TrainDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
            **kwargs,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        **kwargs,
    ) -> DataLoader:
        input_names = get_hybrid_forward_input_names(
            SimpleFeedForwardTrainingNetwork
        )
        instance_splitter = self._create_instance_splitter("validation")
        return ValidationDataLoader(
            dataset=data,
            transform=instance_splitter + SelectFields(input_names),
            batch_size=self.batch_size,
            stack_fn=partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
        )

    # defines the network, we get to see one batch to initialize it. the
    # network should return at least one tensor that is used as a loss to
    # minimize in the training loop. several tensors can be returned for
    # instance for analysis, see DeepARTrainingNetwork for an example.
    def create_training_network(self) -> HybridBlock:
        return SimpleFeedForwardTrainingNetwork(
            num_hidden_dimensions=self.num_hidden_dimensions,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distr_output=self.distr_output,
            batch_normalization=self.batch_normalization,
            mean_scaling=self.mean_scaling,
        )

    # we now define how the prediction happens given that we are provided a
    # training network.
    def create_predictor(self, transformation, trained_network):
        prediction_splitter = self._create_instance_splitter("test")

        if self.sampling is True:
            prediction_network = SimpleFeedForwardSamplingNetwork(
                num_hidden_dimensions=self.num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=self.distr_output,
                batch_normalization=self.batch_normalization,
                mean_scaling=self.mean_scaling,
                params=trained_network.collect_params(),
                num_parallel_samples=self.num_parallel_samples,
            )

            return RepresentableBlockPredictor(
                input_transform=transformation + prediction_splitter,
                prediction_net=prediction_network,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )

        else:
            prediction_network = SimpleFeedForwardDistributionNetwork(
                num_hidden_dimensions=self.num_hidden_dimensions,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=self.distr_output,
                batch_normalization=self.batch_normalization,
                mean_scaling=self.mean_scaling,
                params=trained_network.collect_params(),
                num_parallel_samples=self.num_parallel_samples,
            )
            return RepresentableBlockPredictor(
                input_transform=transformation + prediction_splitter,
                prediction_net=prediction_network,
                batch_size=self.batch_size,
                forecast_generator=DistributionForecastGenerator(
                    self.distr_output
                ),
                prediction_length=self.prediction_length,
                ctx=self.trainer.ctx,
            )