# CL_for_timeseries

Original DAIN github folder with their dataset link : <https://github.com/passalis/dain>

To run GluonTS models, yo have to focus on the main_gluonts.py file and changes the parameters of the estimator as well as capitalized variables.

The code will give an error during dataset split if the time series is shorter than the given context length + prediction length.

In order to run a new GluonTS model, starting from a given GluonTS model, you have to:

- change the _estimator.py, by including in the initializer of the class parameter "mean_layer", that is the learned linear regressor that processes the means, saving them as class attributes, and passing it in the initialization of the training network in the create_training_network method, and in the initialization of the predictor in create_prediction method.
- Add "mean_layer" also in the initialization of the classes in /_network.py file. Generallly speaking, you need only to focus on the initialization of the classes (in which we have to initialize the linear layer that processes the means), and in the hybrid_forward method (that implements the forward pass). You can find the code to do this and to freeze parameters in the initializer of SimpleFeedForwardNetworkBase class in my_simple_feedforward/_network.py
- Include past_feat_dynamic_real in the hybrid_forward of the training network and in the one of the prediction network. Also, include in the Estimator: time_series_field=[FieldName.FEAT_DYNAMIC_REAL, FieldName.OBSERVED_VALUES] in the initialization of the instance splitter (there should be a _create_instance_splitter_method of the estimator)

Be sure to set scaling=False in the initialization of the estimator!
For an example, look at the model in my_simple_feedforward folder.
