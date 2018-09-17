"""Linear model class module

This module contains functions for creating a linear regression model.
"""

import tensorflow

from model import Model


def _get_feature_columns():
    is_working_day_column = tensorflow.feature_column.numeric_column('is_working_day')
    day_of_the_week_column = tensorflow.feature_column.categorical_column_with_identity(
        'day_of_the_week',
        num_buckets=7
    )

    feature_columns = set([
        is_working_day_column,
        day_of_the_week_column
    ])

    return feature_columns


class LinearModel(Model):
    """Class represeting a linear regression model.

    Represents a linear machine learning model that can be trained and can perform prediction
    on given data.
    """

    def train(self, learning_rate, steps, batch_size, training_examples, training_targets,
              validation_examples, validation_targets, regularization_strength=0.0):
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

        optimizer = tensorflow.train.FtrlOptimizer(
            learning_rate=learning_rate,
            l2_regularization_strength=regularization_strength
        )
        optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

        linear_regressor = tensorflow.estimator.LinearRegressor(
            feature_columns=_get_feature_columns(),
            optimizer=optimizer
        )

        super(LinearModel, self)._train_model(
            model=linear_regressor,
            steps=steps,
            batch_size=batch_size,
            training_examples=training_examples,
            training_targets=training_targets,
            validation_examples=validation_examples,
            validation_targets=validation_targets
        )
