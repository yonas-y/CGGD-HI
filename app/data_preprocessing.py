import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple

class DataPreprocessing:
    """
    A class used for cleaning and preprocessing customer satisfaction data.

    Methods
    -------
    data_cleaning() -> pd.DataFrame
        Perform data cleaning procedure such as dropping duplicates, filling missing values,
        and extracting numeric only columns.

    data_encoding(pd.DataFrame) -> pd.DataFrame:
        Encode the non-numeric values of the cleaned data using a OneHotEncoder.

    data_split(pd.DataFrame, test_size, random_state):
        Splits the cleaned and encoded data into training and test sets.

    data_normalization(pd.DataFrame, pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        Normalizes the training and test data!.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The raw input data.
        """
        self.raw_data = dataframe


