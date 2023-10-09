import torch
from torch import Tensor
from torch import nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class FFNN(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        hid_dim_1: int,
        hid_dim_2: int,
        activation: str,
        output_dim: int,
    ) -> None:
        super(FFNN, self).__init__()
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Tanh()
        self.hid_1 = torch.nn.Linear(inp_dim, hid_dim_1)
        self.hid_2 = torch.nn.Linear(hid_dim_1, hid_dim_2)
        self.out_layer = torch.nn.Linear(hid_dim_2, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x is a ts, shape (batch, ts_len, n_features)    # ASSUMING BATCH_FIRST
        x = x.reshape((x.shape[0], -1))
        x = self.activation(self.hid_1(x))
        x = self.activation(self.hid_2(x))
        x = self.out_layer(x)
        return x


from gluonts.mx.model.transformer import TransformerEstimator
from functools import partial
from typing import List, Optional

from gluonts.mx.distribution import DistributionOutput, StudentTOutput
from gluonts.mx.trainer import Trainer
from gluonts.transform import InstanceSampler
from gluonts.time_feature import TimeFeature

from gluonts.mx.model.transformer._network import TransformerTrainingNetwork

class GASTransformerTrainingNetwork(TransformerTrainingNetwork):
class GASTransformerEstimator(TransformerEstimator):
    def __initi____init__(
        self,
        mean_layer: nn.Module,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        trainer: Trainer = Trainer(),
        dropout_rate: float = 0.1,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: int = 20,
        distr_output: DistributionOutput = StudentTOutput(),
        model_dim: int = 32,
        inner_ff_dim_scale: int = 4,
        pre_seq: str = "dn",
        post_seq: str = "drn",
        act_type: str = "softrelu",
        num_heads: int = 8,
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        use_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
        num_parallel_samples: int = 100,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        batch_size: int = 32,
    ) -> None:
        super(GASTransformerEstimator, self).__init__(
            freq,
            prediction_length,
            context_length,
            trainer,
            dropout_rate,
            cardinality,
            embedding_dimension,
            distr_output,
            model_dim,
            inner_ff_dim_scale,
            pre_seq,
            post_seq,
            act_type,
            num_heads,
            scaling,
            lags_seq,
            time_features,
            use_feat_dynamic_real,
            use_feat_static_cat,
            num_parallel_samples,
            train_sampler,
            validation_sampler,
            batch_size,
        )

        self.mean_layer = mean_layer
