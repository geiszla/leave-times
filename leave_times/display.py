"""Data display

This module contains helper functions for plotting and printing data.
"""

import multiprocessing
import sys

from matplotlib import pyplot


# Private methods
def _plot_graph(title, x_label, y_label, data):
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.tight_layout()

    for data_frame in data:
        current_data, current_label = data_frame
        pyplot.plot(current_data, label=current_label)

    pyplot.legend()
    pyplot.show()
    sys.exit()


# Public methods
def plot(title, x_label, y_label, data):
    """
    Plots a graph using the given arguments.

    Args:
        title (str): The title of the plot
        x_label (str): The label of the X-axis
        y_label (str): The label of the Y-axis
        data (numpy.array((
            numpy.array(number): The data points to be plotted,
            str: The corresponding legend for the data points
        ))): An array of tuples of data to be plotted with their legends
    """

    # Start a new process so that the displayed plot doesn't block the execution
    multiprocessing.Process(
        target=_plot_graph,
        args=(title, x_label, y_label, data)
    ).start()


def display_results(training_errors, validation_errors, validation_examples,
                    validation_predictions):
    """
    Pretty prints and plots the result of the model training.

    Args:
        training_errors (numpy.array(float)): An array containing training errors
            at each epoch of training
        validation_errors (numpy.array(float)): An array containing validation errors
            at each epoch of training
        validation_examples (pandas.DataFrame): A data frame containing one or more columns of data
            to use as input features for validation.
        validation_predictions (numpy.array(number)): An array containing predictions
            the model makes on the validation set after training
    """

    plot(
        title='Root Mean Squared Error vs. Periods',
        x_label='Periods',
        y_label='RMSE',
        data=[
            (training_errors, 'training'),
            (validation_errors, 'validation')
        ]
    )

    print('\n==========Validation Examples==========')

    result_table = validation_examples.assign(prediction=validation_predictions)
    print(result_table.to_string())
    print('\nValidation RMSE:', validation_errors[-1])
