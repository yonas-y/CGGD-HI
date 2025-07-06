"""
Project Configuration File

Contains:
- Dataset-specific configuration
- Model architecture hyperparameters
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import tensorflow as tf
import random

# ========= Model architecture hyperparameters ========= #
@dataclass
class ModelHyperparameters:
    input_sample_shape: tuple
    encoding_n: int
    regularization: float
    dropout_rate: float
    pooling_size: int
    kernel_size_conv: int
    stride_convolutional: int
    activation_function: tf.keras.layers.Layer
    kernel_init: tf.keras.initializers.Initializer

# ========= Dataset configuration ========= #
@dataclass
class DatasetConfig:
    """
    Configuration for a specific dataset.
    """
    # Data and Features Directories!
    SETUP_Name: str
    MODEL_OUTPUT_DIR: Path
    FEATURE_DIR: Path

    # Parameters from feature extraction!
    SampleRate: int            # sampling rate of the incoming signal.
    OneSec_Samples: int        # samples representative of 1 sec duration.
    frame_length: int          # number of FFT components.
    hop_length: int            # hop length for spectrogram frames.
    n_mels: int                # number of Mel bands to generate.

    # Parameters for feature preprocessing!
    available_bearings: List[str]  # On which bearing to work on.
    bearing_used: str
    channel: str = 'both'   # Which channel of the features to use. ('vertical', 'horizontal' or 'both').
    n_channels: int = 2     # When "both" is use set it 2, else set it to 1.

    # Dataset-specific optional paths
    TRAIN_RAW_DATA_DIR: Optional[Path] = None
    TEST_RAW_DATA_DIR: Optional[Path] = None
    PICKLE_TRAIN_DIR: Optional[Path] = None
    PICKLE_TEST_DIR: Optional[Path] = None

    model_hyperparams: Optional[ModelHyperparameters] = None
    extra_params: Optional[dict] = None

# ========= Function to build model hyperparameters from dataset config ========= #
def build_model_hyperparams(dataset_cfg: DatasetConfig) -> ModelHyperparameters:
    return ModelHyperparameters(
        input_sample_shape=(dataset_cfg.n_mels, dataset_cfg.n_channels),
        encoding_n=8,
        regularization=1e-3,
        dropout_rate=0.0,
        pooling_size=2,
        kernel_size_conv=3,
        stride_convolutional=2,
        activation_function=tf.keras.layers.ReLU(),
        kernel_init=tf.keras.initializers.GlorotNormal(seed=random.randint(0, 1e6))
    )

# ========= Dataset configurations ========= #
CONFIGS = {
    "pronostia": DatasetConfig(
        SETUP_Name="pronostia",
        MODEL_OUTPUT_DIR=Path("output/pronostia"),
        TRAIN_RAW_DATA_DIR=Path("../../Datasets/Bearings/Pronostia/Dataset/Learning_set/"),
        TEST_RAW_DATA_DIR=Path("../../Datasets/Bearings/Pronostia/Dataset/Full_Test_Set/"),
        PICKLE_TRAIN_DIR=Path("data/raw_pickles/pronostia/training"),
        PICKLE_TEST_DIR=Path("data/raw_pickles/pronostia/test"),
        FEATURE_DIR=Path("data/features/pronostia_mel_features"),
        SampleRate=25600,
        OneSec_Samples=2560,
        frame_length=2560,
        hop_length=2561,
        n_mels=256,
        available_bearings=['Bearing1', 'Bearing2', 'Bearing3'],
        bearing_used='Bearing1',
        channel='both',
        n_channels=2,
        model_hyperparams=None,
    ),
    "XJTU_SY": DatasetConfig(
        SETUP_Name="XJTU_SY",
        MODEL_OUTPUT_DIR=Path("output/XJTU_SY"),
        FEATURE_DIR=Path("data/features/XJTU_SY_mel_features"),
        SampleRate=25600,
        OneSec_Samples=2560,
        frame_length=2560,
        hop_length=2561,
        n_mels=256,
        available_bearings=['Bearing1', 'Bearing2', 'Bearing3', 'Bearing4'],
        bearing_used='Bearing1',
        channel='both',
        n_channels=2,
        model_hyperparams=None,
        extra_params= None
    )
}

# ========= Access function ========= #
def get_config(setup_used: str) -> DatasetConfig:
    cfg = CONFIGS.get(setup_used)
    if cfg is None:
        raise ValueError(f"Unknown SETUP: '{setup_used}'. Available: {list(CONFIGS.keys())}")
    if cfg.model_hyperparams is None:
        cfg.model_hyperparams = build_model_hyperparams(cfg)
    return cfg
