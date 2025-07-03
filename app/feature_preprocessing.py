import pandas as pd
import numpy as np
import joblib
from typing import Tuple

class FeaturePreprocessing:
    """
    A class used for preprocessing the extracted features from Pronostia dataset.

    Methods
    -------
    load_features(): Loads features from the directory containing features.

    scale_features(): Scales features.

    index_features(): 

    split_features(pd.DataFrame, test_size, random_state): Splits features.
        Splits the full run into different sections.


    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The raw input data.
        """
        self.raw_data = dataframe


