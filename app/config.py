"""
Contains Configuration (data, names, paths, .....).
"""

# ========== Directories!!! ============== #
"""
Directories for Data, Features, Model Parameters and ....
"""
MODEL_DIR = "output"
TRAIN_RAW_DATA_DIR = "data/raw/Pronostia/Learning_set"
TEST_RAW_DATA_DIR = "data/raw/Pronostia/Test_set"
PICKLE_TRAIN_DIR = "data/pickles/training"
PICKLE_TEST_DIR = "data/pickles/test"
FEATURE_DIR = "data/features/mel_features"
# MODEL_PATH = f"{MODEL_DIR}/ocsvm_model.pkl"
# SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"


# ========== Parameters from Pronostia dataset! ================== #
"""
Important parameters about the Pronostia dataset for feature extraction.!!
"""
SampleRate = 25600                          # sampling rate of the incoming signal
OneSec_Samples = 2560                         # samples representative of 1 sec duration.
frame_length = 2560                         # number of FFT components
hop_length = 2561                           # hop length for spectrogram frames
n_mels = 256                                # number of Mel bands to generate

# ======== Parameters for feature preprocessing! ================= #
"""
Update these parameters for feature preprocessing and training!
"""
setup='Bearing1'        # On which bearing to work on. ('Bearing1', 'Bearing2' or 'Bearing3')
channel = 'both'        # Which channel of the features to use. ('vertical', 'horizontal' or 'both')
