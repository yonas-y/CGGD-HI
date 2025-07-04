import numpy as np
import joblib
from typing import List, Tuple
from typing_extensions import Annotated
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class feature_preprocessing:
    """
    A class used for preprocessing the extracted features from Pronostia dataset.

    Methods
    -------
    load_features(feature_directory, setup, channel): Loads features from the directory containing features.

    split_scale_features(): Splits the whole data into train and text and scales them.
    """

    def __init__(self, feature_directory: str, setup_used: str, channel_used: str):
        """
        Initialize with the feature directory.

        Args:
            feature_directory: The path to the directory containing features.
            setup_used: which bearing to load!
            channel_used: denotes the channel to use! vertical, horizontal or both!
        """
        self.feature_dir = feature_directory
        self.scaler = StandardScaler()
        self.setup = setup_used
        self.channel = channel_used

    def load_mel_features(self) -> List[np.ndarray]:
        """
        Loads the bearing features!

        """
        feature_dir = Path(self.feature_dir)
        mel_db_feat_list = []

        for file in feature_dir.iterdir():
            if self.setup in file.name:
                # bearing_name = file.name[:10]
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


    def split_scale_features(self, bearing_feature_list: list) -> Tuple[
        Annotated[np.ndarray, "X_train"],
        Annotated[np.ndarray, "X_test"],
        Annotated[np.ndarray, "X_train_scaled"],
        Annotated[np.ndarray, "X_test_scaled"]
    ]:
        """
        Creates train and test sets using the bearing features dictionary.
        Here we use the first two bearings as a training and the remaining as test sets.
        :param bearing_feature_list: Contains a list of bearing run features.
        :return:
        """
        train_features, test_features = bearing_feature_list[:2], bearing_feature_list[2:]
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

        output_dir = Path("output/scaler")
        output_dir.mkdir(exist_ok=True)
        joblib.dump(self.scaler, output_dir / f"{self.setup}_scaler.pkl")  # save

        return train_feat_conc, test_feat_conc, train_scaled, test_scaled
