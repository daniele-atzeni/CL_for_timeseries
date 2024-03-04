from my_models.gluonts_models.univariate.probabilistic_forecast.feedforward_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_linear,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.feedforward_gas_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_gas,
)
from my_models.gluonts_models.univariate.point_forecast.feedforward_gas_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_gas_point,
)
from my_models.gluonts_models.feedforward_multivariate_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_multivariate_linear,
)
from my_models.gluonts_models.feedforward_multivariate_gas_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_multivariate_gas,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.transformer_linear_means._estimator import (
    TransformerEstimator as Transformer_gluonts_linear,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.transformer_gas_means._estimator import (
    TransformerEstimator as Transformer_gluonts_gas,
)
from my_models.gluonts_models.univariate.point_forecast.transformer_gas_means._estimator import (
    TransformerEstimator as Transformer_gluonts_gas_point,
)
from my_models.gluonts_models.transformer_multivariate_linear_means._estimator import (
    TransformerEstimator as Transformer_gluonts_multivariate_linear,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.deepar_gas_means._estimator import (
    DeepAREstimator as Deepar_gluonts_gas,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.deepar_linear_means._estimator import (
    DeepAREstimator as Deepar_gluonts_linear,
)
from my_models.gluonts_models.deepar_multivariate_linear_means._estimator import (
    DeepAREstimator as Deepar_gluonts_multivariate_linear,
)
from my_models.gluonts_models.univariate.wavenet_gas_means._estimator import (
    WaveNetEstimator as Wavenet_gluonts_gas_means,
)

from my_models.pytorch_models.simple_feedforward import FFNN as FF_torch

# dictionary {(name, library, mean_layer, multivariate, probabilistic): model_class}

available_dl_models = {
    ("feedforward", "gluonts", "gas", False, True): FF_gluonts_gas,
    ("feedforward", "gluonts", "gas", False, False): FF_gluonts_gas_point,
    # ("feedforward", "gluonts", "gas", True, True): FF_gluonts_multivariate_gas,
    ("feedforward", "gluonts", "linear", False, True): FF_gluonts_linear,
    # ("feedforward", "gluonts", "linear", True, True): FF_gluonts_multivariate_linear,
    # ("feedforward", "torch", "linear", True, True): FF_torch,
    # ("transformer", "gluonts", "linear", False, True): Transformer_gluonts_linear_means,
    ("transformer", "gluonts", "gas", False, True): Transformer_gluonts_gas,
    (
        "transformer",
        "gluonts",
        "gas",
        False,
        False,
    ): Transformer_gluonts_gas_point,
    # ("transformer", "gluonts", "linear", True, True): Transformer_gluonts_multivariate_linear,
    ("deepar", "gluonts", "gas", False, True): Deepar_gluonts_gas,
    # ("deepar", "gluonts", "linear", False, True): Deepar_gluonts_linear,
    # ("deepar", "gluonts", "gas", True, True): Deepar_gluonts_multivariate_gas,
    # ("deepar", "gluonts", "linear", True, True): Deepar_gluonts_multivariate_linear,
}
