"""Model class module

This module contains the Module class that can be used to construct models.
"""

import math

import numpy
from sklearn import metrics
from tensorflow.python.data import Dataset

from data import restore_category
from display import display_results


def _input_function_delegate(input_features, targets, batch_size=1, is_shuffle=True,
                             epoch_count=None):
    features = {key: numpy.array(value) for key, value in dict(input_features).items()}

    dataset = Dataset.from_tensor_slices((features, targets)).batch(batch_size).repeat(epoch_count)

    if is_shuffle:
        dataset = dataset.shuffle(10000)

    return dataset.make_one_shot_iterator().get_next()


class Model:
    """Abstract class represeting a machine learning model.

    Represents a machine learning model that can be trained and can perform prediction
    on given data. It should be used for base class for specific models
    (e.g. linear model, neural network).
    """

    def __init__(self):
        self.__model = None

    # Public methods
    def predict(self, features, targets):
        """Performs prediction on given values using the trained model.

        Model.train() should be called before using this method.

        Args:
            features (pandas.DataFrame): A data frame containing one or more columns
                of data to use as input features for prediction.
            targets (pandas.DataFrame): A data frame containing one or more columns
                of data to use as target for prediction.

        Returns:
            numpy.array(predictions): The predictions made by the trained model.
        """

        # Function to be used by the model for prediction to get input data
        input_function = lambda: _input_function_delegate(
            features,
            targets,
            is_shuffle=False,
            epoch_count=1
        )

        prediction_results = self.__model.predict(input_fn=input_function)
        training_predictions = numpy.array(
            [result['predictions'][0] for result in prediction_results]
        )

        return training_predictions

    # Protected methods
    def _train_model(self, model, steps, batch_size, training_examples, training_targets,
                     validation_examples, validation_targets):
        self.__model = model

        # Function to be used by the model at training to get input data
        training_input_function = lambda: _input_function_delegate(
            training_examples,
            training_targets,
            batch_size=batch_size
        )

        # Training
        print('==========Training==========')
        print("RMS error on training data:")

        # Partition steps to periods to print partial result while running
        periods = 10
        steps_per_period = steps / periods

        training_errors = []
        validation_errors = []
        for period in range(0, periods):
            self.__model.train(input_fn=training_input_function, steps=steps_per_period)

            # Test model on training data
            training_predictions = self.predict(training_examples, training_targets)

            # Calculate training error
            training_error = math.sqrt(
                metrics.mean_squared_error(training_predictions, training_targets)
            )
            training_errors.append(training_error)
            print('  period %02d : %0.2f' % (period, training_error))

            # Test model on validation data
            validation_predictions = self.predict(validation_examples, validation_targets)
            last_validation_predctions = validation_predictions

            # Calculate validation error
            validation_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions, validation_targets)
            )
            validation_errors.append(validation_error)

        # Restore original values of categories and display result
        restored_examples = restore_category(validation_examples)
        display_results(
            training_errors,
            validation_errors,
            restored_examples,
            last_validation_predctions
        )
