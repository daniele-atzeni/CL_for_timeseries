import pandas as pd

from gluonts.dataset.repository import get_dataset

from gluonts.mx import SimpleFeedForwardEstimator, Trainer

from gluonts.dataset.field_names import FieldName

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AdhocTransform,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
)


def create_transformation(freq, context_length, prediction_length, myfn):
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=1,
                    min_future=prediction_length,
                ),
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[
                    FieldName.FEAT_AGE,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
            AdhocTransform(myfn),
        ]
    )


dataset = get_dataset("m4_hourly")
print(type(dataset))

train_entry = next(iter(dataset.train))

trainer = Trainer(
    ctx="cpu",
    epochs=5,
    learning_rate=1e-3,
    hybridize=False,
    num_batches_per_epoch=100,
)

estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=2 * dataset.metadata.prediction_length,
    trainer=trainer,
)


def myfn(x):
    x["old"] = x["future_target"].copy()
    x["future_target"] = x["future_target"] * 2
    return x


transformation = create_transformation(
    dataset.metadata.freq,
    2 * dataset.metadata.prediction_length,  # can be any appropriate value
    dataset.metadata.prediction_length,
    myfn,
)

train_ds = dataset.train

train_tf = transformation(iter(train_ds), is_train=True)

train_tf_entry = next(iter(train_tf))

print(f"past target shape: {train_tf_entry['past_target'].shape}")
print(f"future target shape: {train_tf_entry['future_target'].shape}")
print(f"past observed values shape: {train_tf_entry['past_observed_values'].shape}")
print(f"future observed values shape: {train_tf_entry['future_observed_values'].shape}")
print(f"past age feature shape: {train_tf_entry['past_feat_dynamic_age'].shape}")
print(f"future age feature shape: {train_tf_entry['future_feat_dynamic_age'].shape}")
print(train_tf_entry["feat_static_cat"])

########### future_target field to calculate the error of our predictions

print("end")
