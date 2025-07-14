from zenml import step
import numpy as np
from typing import List, Tuple, Annotated
from pathlib import Path
from app.materializers.joblib_list_materializer import JoblibListMaterializer

from app.feature_extraction import feature_extraction
from app.feature_preprocessing import feature_preprocessing, features_ene_rul_train
from app.feature_partitioning import create_feature_portions, shuffle_batched_interleaved

from app.active_config import cfg

import logging
logger = logging.getLogger(__name__)

@step(enable_cache=False)
def feature_extraction_step(pickle_dir: Path, feature_dir: Path) -> None:
    """
    Step to extract features from the input pickle data.
    """
    logger.info(f"📦 Extracting features ...")
    feature_extraction(pickle_dir, feature_dir)

    return

@step(enable_cache=False,
      output_materializers={
          "train_data_scaled": JoblibListMaterializer,
          "train_Ene_RUL_Order": JoblibListMaterializer,
          "test_data_scaled": JoblibListMaterializer
      }
      )
def feature_preprocessing_step(feature_directory: Path, output_directory: Path,
                               bearing_used: str, channel_used: str) -> Tuple[
    Annotated[List[np.ndarray], "train_data_scaled"],
    Annotated[List[np.ndarray], "train_Ene_RUL_Order"],
    Annotated[List[np.ndarray], "test_data_scaled"]
]:
    """
    Load mel features from the feature directory, then split and scale them
    into train/test sets.

    return:
     X_train_scaled, Ene_RUL_Order_train, X_test_scaled
    """
    # Create the preprocessing class!
    logger.info("Initializing feature preprocessing...")
    feature_preprocess = feature_preprocessing(feature_directory, output_directory, bearing_used, channel_used)

    # Load the features!
    logger.info("Loading mel features...")
    feature_db_list = feature_preprocess.load_mel_features(cfg.n_mels)

    if not feature_db_list:
        raise ValueError("No features loaded; check your feature directory or config.")

    # Split and scale the features!
    logger.info("Splitting and scaling features...")
    train_index_list = cfg.bearing_splits[bearing_used]["train_index"]
    test_index_list = cfg.bearing_splits[bearing_used]["test_index"]
    X_train, X_test, X_train_scaled, X_test_scaled = feature_preprocess.split_scale_features(
        feature_db_list, train_index_list, test_index_list)

    # Extract the scaled energy, RUL and order of the training samples!
    logger.info("Calculating the feature energy ...")
    Ene_RUL_order_train = features_ene_rul_train(X_train)

    logger.info("Feature preprocessing completed successfully.")
    return X_train_scaled, Ene_RUL_order_train, X_test_scaled

@step(enable_cache=False,
      output_materializers={
          "X_train_lists": JoblibListMaterializer
      }
      )
def feature_partitioning_step(feature_mel: List[np.ndarray],
                              feature_ene_rul_order: List[np.ndarray],
                              percentages: List[float]) -> Annotated[List[List[np.ndarray]], "X_train_lists"]:
    # Partitions the features into start, mid and final sections segments of the run!
    X_train_lists = create_feature_portions(feature_mel, feature_ene_rul_order, percentages)

    return X_train_lists

@step(enable_cache=False,
      output_materializers={
          "training_data": JoblibListMaterializer,
          "validation_data": JoblibListMaterializer
      }
      )
def train_validation_split_step(
        percentage_partitioned_data: List[List[np.ndarray]],
        batch_percentages: List[float],
        val_split: float,
        batch_size: int
)-> Tuple[
    Annotated[List, "training_data"],
    Annotated[List, "validation_data"]]:

    # Shuffles and partitions the data into training and validation sets!
    training_data, validation_data = shuffle_batched_interleaved(percentage_partitioned_data,
                                                                 batch_percentages,
                                                                 val_split, batch_size)

    return training_data, validation_data
