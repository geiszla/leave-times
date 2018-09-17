"""Data processing and preparation

This module contains data-specific functions to preprocess and prepare the data
for a machine learning model to use.
"""

import pandas
from sklearn.preprocessing import LabelEncoder

LABEL_ENCODER = None


# Private methods
def _preprocess_features(dataframe):
    processed_features = dataframe[[
        'is_working_day',
    ]].copy()

    # Encode day of the week values as integers
    global LABEL_ENCODER
    LABEL_ENCODER = LabelEncoder()

    integer_encoded = LABEL_ENCODER.fit_transform(dataframe['day_of_the_week'])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # one_hot_encoded = OneHotEncoder(sparse=True).fit_transform(integer_encoded)
    processed_features['day_of_the_week'] = integer_encoded

    return processed_features


def _preprocess_targets(dataframe):
    processed_targets = pandas.DataFrame()
    processed_targets['leave_minutes'] = dataframe['leave_minutes'].copy()

    return processed_targets


# Public methods
def preprocess_data(dataframe):
    """Preprocesses the leave-times data-set and produce the feature and target data frames.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing raw data imported
            from e.g. a csv file.

    Returns:
        (
            pandas.DataFrame: The preprocessed features dataframe,
            pandas.DataFrame: The preprocessed targets dataframe
        )
    """

    return _preprocess_features(dataframe), _preprocess_targets(dataframe)

def restore_category(dataframe):
    """Restores the original values of category columns.

    Args:
        dataframe (pandas.DataFrame): The dataframe containing raw data imported
            from e.g. a csv file.

    Returns:
        pandas.DataFrame: The dataframe with restored category values
    """

    restored_features = dataframe.copy()
    restored_features['day_of_the_week'] = LABEL_ENCODER.inverse_transform(
        dataframe['day_of_the_week']
    )

    return restored_features
