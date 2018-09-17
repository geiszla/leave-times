"""Neural network class module

This module contains functions for creating a deep neural network regression model.
"""

import tensorflow

from model import Model


def _get_feature_columns():
    is_working_day_column = tensorflow.feature_column.numeric_column('is_working_day')
    day_of_the_week_column = tensorflow.feature_column.embedding_column(
        tensorflow.feature_column.categorical_column_with_identity(
            'day_of_the_week',
            num_buckets=7
        ),
        dimension=7
    )

    feature_columns = set([
        is_working_day_column,
        day_of_the_week_column
    ])

    return feature_columns


class NeuralNetworkModel(Model):
    """Class represeting a deep neural network regression model.

    Represents a neural network model that can be trained and can perform prediction
    on given data.
    """

    def __init__(self, hidden_units):
        super(NeuralNetworkModel, self).__init__()
        self.__hidden_units = hidden_units

    def train(self, learning_rate, steps, batch_size, training_examples, training_targets,
              validation_examples, validation_targets):
        """
        Trains the model using the given arguments and displays the result.

        Args:
            steps (int; non-zero): The total number of training steps. A training step
                consists of a forward and backward pass using a single batch.
            batch_size (int; non-zero): Size of one batch on which optimization is run
            training_examples (pandas.DataFrame): A data frame containing one or more columns
                of data to use as input features for training.
            training_targets (pandas.DataFrame): A data frame containing one or more columns
                of data to use as target for training.
            validation_examples (pandas.DataFrame): A data frame containing one or more columns
                of data to use as input features for validation.
            validation_targets (pandas.DataFrame): A data frame containing one or more columns
                of data to use as target for validation.
        """

        optimizer = tensorflow.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

        dnn_regressor = tensorflow.estimator.DNNRegressor(
            feature_columns=_get_feature_columns(),
            hidden_units=self.__hidden_units,
            optimizer=optimizer
        )

        super(NeuralNetworkModel, self)._train_model(
            model=dnn_regressor,
            steps=steps,
            batch_size=batch_size,
            training_examples=training_examples,
            training_targets=training_targets,
            validation_examples=validation_examples,
            validation_targets=validation_targets
        )
