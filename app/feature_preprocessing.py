import os
import numpy as np
import joblib
from typing import List, Tuple, Dict
from typing_extensions import Annotated
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
import librosa

def features_ene_rul_train(train_feature_list: list) -> List[np.ndarray]:
    """
    Calculated the scaled energy of the training samples in the run and their location in the run.
    In addition, the RUL is assumed to be decreasing from 1 to 0 through the run.

    :param train_feature_list: mel training feature list of the bearing under consideration.
    :return: A list where each entry contains a stacked array with scaled energy, rul, and order.
    """
    combined_feat_list = []

    for i in range(len(train_feature_list)):
        mel_feature = train_feature_list[i]
        # Compute mean energy per sample (over mel bands and channels)
        mel_band_energies_mean = np.mean(mel_feature ** 2, axis=(1, 2))

        # Convolve the energy to remove sharp fluctuations! Smooth with moving average (window size = 12)
        mel_band_energies_smooth = uniform_filter1d(mel_band_energies_mean, size=12, mode='nearest')

        # Convert to dB scale.
        mel_band_energies_mean_db = librosa.power_to_db(mel_band_energies_smooth, ref=np.median)

        # Scale the energy!
        min_val = np.min(mel_band_energies_mean_db)
        max_val = np.max(mel_band_energies_mean_db)
        mel_band_energies_mean_scaled = (mel_band_energies_mean_db - min_val) / (max_val - min_val)

        # Calculate a decreasing RUL and numerical ordering of the samples in time!
        RUL = np.linspace(1.0, 0.0, num=mel_feature.shape[0])
        order = np.array(range(1, mel_feature.shape[0] + 1))

        combined = np.stack([mel_band_energies_mean_scaled, RUL, order], axis=1)

        combined_feat_list.append(combined)

    return combined_feat_list


class feature_preprocessing:
    """
    A class used for preprocessing the extracted features from pronostia dataset.

    Methods
    -------
    load_features(feature_directory, bearing, channel): Loads features from the directory containing features.

    split_scale_features(): Splits the whole data into train and text and scales them.
    """

    def __init__(self, feature_directory: Path,
                 output_directory: Path,
                 bearing_used: str,
                 channel_used: str):
        """
        Initialize with the feature directory.

        Args:
            feature_directory: The path to the directory containing features.
            output_directory: The output path where to store different step outputs!
            bearing_used: which bearing to load!
            channel_used: denotes the channel to use! vertical, horizontal or both!
        """
        self.feature_dir = feature_directory
        self.output_dir = output_directory
        self.scaler = StandardScaler()
        self.bearing = bearing_used
        self.channel = channel_used

    def load_mel_features(self, n_mels) -> List[np.ndarray]:
        """
        Loads the bearing features!

        """
        mel_db_feat_list = []

        for file in self.feature_dir.iterdir():
            if(
                    file.is_file() and
                    file.name.startswith(self.bearing) and
                    file.name.endswith(f"{str(n_mels)}.npy")
            ):
                mel_db_feat = np.load(file)

                if self.channel == 'horizontal':
                    mel_db_feat = mel_db_feat[:, :, 1:]
                elif self.channel == 'vertical':
                    mel_db_feat = mel_db_feat[:, :, :1]
                elif self.channel == 'both':
                    mel_db_feat = mel_db_feat
                else:
                    print("Use either 'horizontal' or 'vertical' or 'both'!")

                mel_db_feat_list.append(mel_db_feat)

        return mel_db_feat_list


    def split_scale_features(self, bearing_feature_list: list, train_indexes: list, test_indexes: list) -> Tuple[
        List[np.ndarray],   # "X_train"
        List[np.ndarray],   # "X_test"
        List[np.ndarray],   # "X_train_scaled"
        List[np.ndarray],   # "X_test_scaled"
    ]:
        """
        Creates train and test sets using the bearing features dictionary.
        Here we use the first two bearings as a training and the remaining as test sets.
        :param bearing_feature_list: Contains a list of bearing run features.
        :param train_indexes: A list of the indexes of the training samples.
        :param test_indexes: A list of the indexes of the test samples.
        :return: The train and test sets and also their scaled values.
        """
        train_features = [bearing_feature_list[i] for i in train_indexes]
        test_features = [bearing_feature_list[i] for i in test_indexes]

        # Keep track of original lengths
        train_lengths = [arr.shape[0] for arr in train_features]
        test_lengths = [arr.shape[0] for arr in test_features]

        train_feat_conc = np.concatenate(train_features)
        test_feat_conc = np.concatenate(test_features)

        # Flatten first two dimensions
        # Shape: (# of samples, mel bands, channel)
        train_reshaped = train_feat_conc.reshape(-1, train_feat_conc.shape[2])
        test_reshaped = test_feat_conc.reshape(-1, test_feat_conc.shape[2])

        train_features_scaled = self.scaler.fit_transform(train_reshaped)
        test_features_scaled = self.scaler.transform(test_reshaped)

        # Reshape back to original 3D shapes
        train_scaled = train_features_scaled.reshape(train_feat_conc.shape)
        test_scaled = test_features_scaled.reshape(test_feat_conc.shape)

        # Split back to original list structure
        train_split_indices = np.cumsum(train_lengths)[:-1]
        test_split_indices = np.cumsum(test_lengths)[:-1]

        train_scaled_list = np.split(train_scaled, train_split_indices, axis=0)
        test_scaled_list = np.split(test_scaled, test_split_indices, axis=0)

        scaler_dir = Path(self.output_dir) / "scaler"  # now scaler_dir is a Path
        scaler_dir.mkdir(parents=True, exist_ok=True)  # safely create directory if needed
        joblib.dump(self.scaler, scaler_dir / f"{self.bearing}_scaler.pkl")  # save the scaler!

        return train_features, test_features, train_scaled_list, test_scaled_list
