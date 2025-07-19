"""
Project Configuration File

Contains:
- Dataset-specific configuration
- Model architecture hyperparameters
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
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
    stride_conv: int
    activation_function: tf.keras.layers.Layer
    kernel_init: tf.keras.initializers.Initializer
    bias_init: tf.keras.initializers.Initializer

# ========= Model training parameters ========= #
@dataclass
class ModelTrainingParameters:
    # Parameters for run feature partitioning!
    run_partitioning_portion: tuple
    segment_percentages_in_batch: tuple

    batch_size: int
    validation_split: float
    epochs: int
    patience: int
    training_iterations: int
    save_weights: bool
    early_stop: bool

# =========== Constraint Parameters ============== #
@dataclass
class ConstraintParameters:
    """
    Configuration for constraint hyperparameters used in the model.
    """
    # Rescale factors (rf) for different constraints
    reconstruction_rf: float
    soft_rank_rf: float
    monotonicity_rf: tuple
    energy_hi_dev_rf: float
    upper_bound_rf: float
    lower_bound_rf: float

    # Bounds for the two extreme health indicator cases
    max_cutoff: float
    upper_cutoff: float
    lower_cutoff: float
    min_cutoff: float

    # Bounds for the normalized energy!
    min_scaled_energy: float
    max_scaled_energy: float

    # Regularization factor for Spearman's rank correlation
    spearmans_regularization: float

# ========= Dataset configuration ========= #
@dataclass
class DatasetConfig:
    """
    Configuration for a specific dataset.
    """
    # Data and Features Directories!
    SETUP_Name: str
    model_type: str
    OUTPUT_DIR: Path
    SETUP_RAW_DIRS: list  # list of Path objects to the different folders of each setup!
    PICKLE_DATA_DIR: Path
    FEATURE_DIR: Path

    # Parameters from feature extraction!
    SampleRate: int            # sampling rate of the incoming signal.
    frame_length: int          # Collected continuous samples per frame.
    n_fft: int                 # smaller n_fft → better time resolution
    hop_length: int            # smaller hop_length → more frames
    n_mels: int                # number of Mel bands to generate.

    # Parameters for feature preprocessing!
    bearing_splits: Dict[str, Dict[str, List[str]]] #  train/test split info
    bearing_used: str
    channel: str    # Which channel of the features to use. ('vertical', 'horizontal' or 'both').

    @property
    def n_channels(self):
        return 2 if self.channel == 'both' else 1

    model_hyperparams: Optional[ModelHyperparameters] = None
    constraint_params: Optional[ConstraintParameters] = None
    model_training_params: Optional[ModelTrainingParameters] = None
    extra_params: Optional[dict] = None

# ========= Function to build model hyperparameters from dataset config ========= #
def build_model_hyperparams(dataset_cfg: DatasetConfig) -> ModelHyperparameters:
    return ModelHyperparameters(
        input_sample_shape=(dataset_cfg.n_mels, dataset_cfg.n_channels),
        encoding_n=16,
        regularization=1e-3,
        dropout_rate=0.0,
        pooling_size=2,
        kernel_size_conv=3,
        stride_conv=1,
        activation_function=tf.keras.layers.ReLU(),
        kernel_init=tf.keras.initializers.GlorotNormal(seed=random.randint(0, 1e6)),
        bias_init=tf.keras.initializers.GlorotNormal(seed=random.randint(0, 1e6))

    )

# ========= Function to build model training parameters ========= #
def build_model_training_params() -> ModelTrainingParameters:
    # Use dataset_cfg to adjust params if needed, or set static defaults
    return ModelTrainingParameters(
        run_partitioning_portion=(0.1, 0.85, 0.05),
        segment_percentages_in_batch=(0.2, 0.7, 0.1),
        batch_size=64,
        validation_split=0.1,
        epochs=150,
        patience=15,
        training_iterations=10,
        save_weights=False,
        early_stop=False
    )

# ========= Function to build constraint parameters from dataset config ========= #
def build_constraint_params() -> ConstraintParameters:
    # Use dataset_cfg to adjust params if needed, or set static defaults
    return ConstraintParameters(
        reconstruction_rf=1.0,
        soft_rank_rf=0.0,
        monotonicity_rf=(0.0, 0.01),
        energy_hi_dev_rf=0.0,
        upper_bound_rf=0.0,
        lower_bound_rf=0.0,
        max_cutoff=1.0,
        upper_cutoff=0.9,
        lower_cutoff=0.1,
        min_cutoff=0.0,
        spearmans_regularization=0.1,
        max_scaled_energy=1.0,
        min_scaled_energy=0.0
    )

def get_config(setup_name: str, Local: bool) -> DatasetConfig:
    if setup_name == "pronostia":
        if Local:
            raw_dirs = [
                Path("../../Datasets/Bearings/Pronostia/Dataset/Learning_set/"),
                Path("../../Datasets/Bearings/Pronostia/Dataset/Full_Test_Set/")
            ]
            pickle_dir = Path("../../Datasets/Bearings/raw_pickles/pronostia/")
            out_dir = Path("output/")
        else:
            raw_dirs = [
                Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/2012-PRONOSTIA/Learning_set/"),
                Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/2012-PRONOSTIA/Full_Test_Set/")
            ]
            pickle_dir = Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/yonas/Datasets/Bearings/raw_pickles/pronostia/")
            out_dir = Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/yonas/CheckPoints/")

        cfg = DatasetConfig(
            SETUP_Name="pronostia",
            model_type="CCAE",
            OUTPUT_DIR=out_dir,
            SETUP_RAW_DIRS=raw_dirs,
            PICKLE_DATA_DIR=pickle_dir,
            FEATURE_DIR=Path("data/features/pronostia_mel_features"),
            SampleRate=25600,
            frame_length=2560,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            bearing_used='Bearing1',
            channel='both',
            bearing_splits={
                "Bearing1": {"train_index": [0, 1], "test_index": [2, 3, 4, 5, 6]},
                "Bearing2": {"train_index": [0, 1], "test_index": [2, 3, 4, 5, 6]},
                "Bearing3": {"train_index": [0, 1], "test_index": [2]}
            },
            model_hyperparams=None,
            constraint_params=None,
            model_training_params=None,
            extra_params=None
        )

    elif setup_name == "XJTU_SY":
        if Local:
            raw_dirs = [
                Path("../../Datasets/Bearings/XJTU-SY_Bearing_Datasets/Data/35Hz12kN/"),
                Path("../../Datasets/Bearings/XJTU-SY_Bearing_Datasets/Data/37_5Hz11kN/"),
                Path("../../Datasets/Bearings/XJTU-SY_Bearing_Datasets/Data/40Hz10kN/")
            ]
            pickle_dir = Path("../../Datasets/Bearings/raw_pickles/XJTU_SY/")
            out_dir = Path("output/XJTU_SY")
        else:
            raw_dirs = [
                Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/XJTU-SY_Bearing_Datasets/Data/35Hz12kN/"),
                Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/XJTU-SY_Bearing_Datasets/Data/37_5Hz11kN/"),
                Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/XJTU-SY_Bearing_Datasets/Data/40Hz10kN/")
            ]
            pickle_dir = Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/yonas/Datasets/Bearings/raw_pickles/XJTU_SY/")
            out_dir = Path("/cw/dtaidata/dtaigeel/NoCsBack/dtai/yonas/CheckPoints/")

        cfg = DatasetConfig(
            SETUP_Name="XJTU_SY",
            model_type="CCAE",
            OUTPUT_DIR=out_dir,
            SETUP_RAW_DIRS=raw_dirs,
            PICKLE_DATA_DIR=pickle_dir,
            FEATURE_DIR=Path("data/features/XJTU_SY_mel_features"),
            SampleRate=25600,
            frame_length=32768,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            bearing_used='Bearing1',
            channel='both',
            bearing_splits={
                "Bearing1": {"train_index": [1, 3, 4], "test_index": [0, 2]},
                "Bearing2": {"train_index": [0, 1, 2], "test_index": [3, 4]},
                "Bearing3": {"train_index": [1, 3, 4], "test_index": [0, 2]}
            },
            model_hyperparams=None,
            constraint_params=None,
            model_training_params=None,
            extra_params=None
        )
    else:
        raise ValueError(f"Unknown setup: {setup_name}")

    cfg.model_hyperparams = build_model_hyperparams(cfg)
    cfg.model_training_params = build_model_training_params()
    cfg.constraint_params = build_constraint_params()

    return cfg

# ========= Parameters update function ========= #
def update_config(cfg: DatasetConfig,
                  model_type: str = "CCAE",
                  bearing_used: Optional[str] = None,
                  channel: Optional[str] = None,
                  **extra_params):
    """
    Update selected fields of DatasetConfig dynamically.
    """
    if cfg.model_type is not None:
        cfg.model_type = model_type
    if bearing_used is not None:
        cfg.bearing_used = bearing_used
    if channel is not None:
        cfg.channel = channel

    # Optionally update anything else dynamically
    for key, value in extra_params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise AttributeError(f"DatasetConfig has no attribute '{key}'")
