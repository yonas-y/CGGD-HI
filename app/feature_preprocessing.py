import pandas as pd
import numpy as np
import joblib
from typing import Dict
from app.config import setup, channel
from pathlib import Path

class FeaturePreprocessing:
    """
    A class used for preprocessing the extracted features from Pronostia dataset.

    Methods
    -------
    load_features(feature_directory, setup, channel): Loads features from the directory containing features.

    scale_features(): Scales features.

    index_features():

    split_features(pd.DataFrame, test_size, random_state): Splits features.
        Splits the full run into different sections.

    """

    def __init__(self, feature_directory):
        """
        Initialize with a pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The raw input data.
        """
        self.feature_dir = feature_directory

    def load_mel_features(self, setup_used=setup, channel_used=channel):
        """
        Loads the bearing features!
        :param setup_used: which bearing to load!
        :param channel_used: denotes the channel to use! vertical, horizontal or both!
        """
        feature_dir = Path(self.feature_dir)
        mel_db_feat_runs_dict = {}

        for file in feature_dir.iterdir():
            if setup_used in file.name:
                bearing_name = file.name[:10]
                mel_db_feat = np.load(file)

                if channel_used == 'horizontal':
                    mel_db_feat = mel_db_feat[:, :, 1:]
                elif channel_used == 'vertical':
                    mel_db_feat = mel_db_feat[:, :, :1]
                elif channel_used == 'both':
                    mel_db_feat = mel_db_feat
                else:
                    print("Use either 'horizontal' or 'vertical' or 'both'!")

                mel_db_feat_dict[bearing_name] = mel_db_feat

        return mel_db_feat_dict
