from pathlib import Path
import pandas as pd
import numpy as np
from app.active_config import cfg
from app.time_series_features import signal_features


def feature_extraction(PICKLE_DIR: Path, FEATURE_OUT_DIR: Path) -> None:
    """
    Can extract multiple features from the time series data and store it as a numpy array in
    the feature directory.

    :param PICKLE_DIR: The directory containing the time series data stored as a pickle file.
    :param FEATURE_OUT_DIR: The directory to store the feature arrays.

    :return:
    """

    for file in PICKLE_DIR.glob('Bearing*.pkl'):

        output_feature = FEATURE_OUT_DIR / f"{file.stem[:10]}_feat_mel_DB_{cfg.n_mels}.npy"

        # Check weather it is already extracted or not!
        if output_feature.exists():
            print(f"[✓] Skipping {file.name} — already extracted!")
            continue

        print(f"[→] Extracting the feature for {file.name}")
        try:
            bearing_data = pd.read_pickle(file)
        except Exception as e:
            print(f"Failed to read pickle file {file.name}: {e}")
            continue

        # Extract relevant columns safely
        try:
            bearing_data_V = bearing_data['Vert. accel.'].to_numpy()
            bearing_data_H = bearing_data['Horiz. accel.'].to_numpy()
        except KeyError as e:
            print(f"Missing expected column in {file.name}: {e}")
            continue

        ACM_V_Feat_list, ACM_H_Feat_list = [], []
        for sample in range(0, len(bearing_data), cfg.frame_length):
            ACM_V_sample = bearing_data_V[sample:sample + cfg.frame_length]
            ACM_H_sample = bearing_data_H[sample:sample + cfg.frame_length]

            # Vertical and Horizontal classes!!
            feat_class_V = signal_features(ACM_V_sample, cfg.SampleRate, cfg.n_fft, cfg.hop_length)
            feat_class_H = signal_features(ACM_H_sample, cfg.SampleRate, cfg.n_fft, cfg.hop_length)

            """
            Possible to extract features and use it later for model input from the 
            package time_series_features.py!

            Like:
            # # Max and Min values of the frame!
            # max_value_V, min_value_V = feat_class_V.signal_max_and_min()
            # max_value_H, min_value_H = feat_class_H.signal_max_and_min()
            """

            # Mel spectral matrices of the frame!
            _, mel_S_dB_V = feat_class_V.signal_mel_spectrogram(N_mels=cfg.n_mels)
            _, mel_S_dB_H = feat_class_H.signal_mel_spectrogram(N_mels=cfg.n_mels)

            # Aggregate over time axis
            mel_S_dB_V = np.mean(mel_S_dB_V, axis=1)
            mel_S_dB_H = np.mean(mel_S_dB_H, axis=1)

            # Collect flattened features into lists
            ACM_V_Feat_list.append(mel_S_dB_V.ravel())
            ACM_H_Feat_list.append(mel_S_dB_H.ravel())

        # Convert to numpy arrays
        ACM_V_Feat = np.array(ACM_V_Feat_list)[..., np.newaxis]  # shape: (n_samples, n_features, 1)
        ACM_H_Feat = np.array(ACM_H_Feat_list)[..., np.newaxis]

        # Concatenate along the last axis to get combined features
        ACM_Feat = np.concatenate((ACM_V_Feat, ACM_H_Feat), axis=2)  # shape: (n_samples, n_features, 2)

        # Save the features for later use as a numpy file!
        np.save(output_feature, ACM_Feat)

    return