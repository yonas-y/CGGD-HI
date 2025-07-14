from zenml import pipeline
from typing import Tuple, List
from steps.data_import_step import import_and_catch_data_step
from steps.feature_extraction_preprocessing_step import (feature_extraction_step,
                                                         feature_preprocessing_step,
                                                         feature_partitioning_step,
                                                         train_validation_split_step)
from app.active_config import cfg

@pipeline
def data_pipeline() -> Tuple[List, List, List]:
    # Raw data importing step!
    for raw_data_dir in cfg.SETUP_RAW_DIRS:
        # Import raw data from directories!
        import_and_catch_data_step(setup_name = cfg.SETUP_Name,
                                   raw_data_dir = raw_data_dir,
                                   pickle_dir = cfg.PICKLE_DATA_DIR)

    # Feature extraction step!
    feature_extraction_step(pickle_dir = cfg.PICKLE_DATA_DIR,
                            feature_dir = cfg.FEATURE_DIR)

    # Feature Preprocessing step!
    X_train_scaled, Ene_RUL_order_train, X_test_scaled = feature_preprocessing_step(
        feature_directory=cfg.FEATURE_DIR,
        output_directory=cfg.OUTPUT_DIR,
        bearing_used=cfg.bearing_used,
        channel_used=cfg.channel
    )

    # Partitioning a run into different segments step! (Start, Middle, Last)
    feature_partitioned_lists = (
        feature_partitioning_step(feature_mel=X_train_scaled,
                                  feature_ene_rul_order = Ene_RUL_order_train,
                                  percentages = cfg.model_training_params.run_partitioning_portion))

    # Split into training and validation sets step!
    training_scaled_data, validation_scaled_data = train_validation_split_step(
        percentage_partitioned_data=feature_partitioned_lists,
        batch_percentages=cfg.model_training_params.segment_percentages_in_batch,
        val_split=cfg.model_training_params.validation_split,
        batch_size=cfg.model_training_params.batch_size )

    return training_scaled_data, validation_scaled_data, X_test_scaled
