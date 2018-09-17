"""Main module

This module trains a machine learning model for the leave time data set.

Todo:
     - One hot encoding for day_of_the_week
"""

import math

import numpy
import pandas
from matplotlib import pyplot

from data import preprocess_data, restore_category
# from linear_regression import create_model
from neural_network import NeuralNetworkModel


def _init():
    # Read data and shuffle it
    dataframe = pandas.read_csv('data/log.csv', sep=',', header=0)
    dataframe = dataframe.reindex(numpy.random.permutation(dataframe.index))

    # Generate features and targets
    processed_features, processed_targets = preprocess_data(dataframe)

    print('==========Feature Summary==========')
    print(processed_features.describe())
    print()
    print('==========Target Summary===========')
    print(processed_targets.describe())
    print()

    # ratio = training data / all data
    training_data_ratio = 0.8

    # Separate data to training and validation sets
    training_data_count = math.ceil(training_data_ratio * len(dataframe))
    training_examples = processed_features.head(training_data_count)
    training_targets = processed_targets.head(training_data_count)

    validation_data_count = len(dataframe) - training_data_count
    validation_examples = processed_features.tail(validation_data_count)
    validation_targets = processed_targets.tail(validation_data_count)

    # Train neural network
    model = NeuralNetworkModel(hidden_units=[10, 10])
    model.train(
        learning_rate=0.005,
        steps=750,
        batch_size=training_data_count,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets
    )

    # Predict unlabeled data and display result
    unlabeled_dataframe = pandas.read_csv('data/log_unlabeled.csv', sep=',', header=0)
    unlabeled_features, unlabeled_targets = preprocess_data(unlabeled_dataframe)
    unlabeled_predictions = model.predict(unlabeled_features, unlabeled_targets)

    result_table = restore_category(unlabeled_features).assign(prediction=unlabeled_predictions)
    print('\n==========Unlabeled Data Predictions==========')
    print(result_table.to_string())


if __name__ == '__main__':
    _init()
