from zenml import step
import os
import logging
import numpy as np
from typing import List, Tuple
from pathlib import Path

from app.feature_extraction import feature_extraction
from app.feature_preprocessing import feature_preprocessing
from app.feature_preprocessing import features_ene_rul_train

logger = logging.getLogger(__name__)

@step(enable_cache=False)
def feature_extraction_step(pickle_dir: Path, feature_dir: Path) -> None:
    """
    Step to extract features from the input pickle data.
    """
    logger.info(f"📦 Extracting features ...")
    feature_extraction(pickle_dir, feature_dir)

    return

@step(enable_cache=False)
def feature_preprocessing_step(feature_directory: Path, output_directory: Path,
                               bearing_used: str, channel_used: str) -> Tuple[
    List[np.ndarray],       # X_train_scaled
    List[np.ndarray],       # Ene_RUL_Order_train
    List[np.ndarray]        # X_test_scaled
]:
    """
    Load mel features from the feature directory, then split and scale them
    into train/test sets.
    """
    # Create the preprocessing class!
    logger.info("Initializing feature preprocessing...")
    feature_preprocess = feature_preprocessing(feature_directory, output_directory, bearing_used, channel_used)

    # Load the features!
    logger.info("Loading mel features...")
    feature_db_list = feature_preprocess.load_mel_features()

    if not feature_db_list:
        raise ValueError("No features loaded; check your feature directory or config.")

    # Split and scale the features!
    logger.info("Splitting and scaling features...")
    X_train, X_test, X_train_scaled, X_test_scaled = feature_preprocess.split_scale_features(feature_db_list)

    # Extract the scaled energy, RUL and order of the training samples!
    logger.info("Calculating the feature energy ...")
    Ene_RUL_order_train = features_ene_rul_train(X_train)

    logger.info("Feature preprocessing completed successfully.")
    return X_train_scaled, Ene_RUL_order_train, X_test_scaled
