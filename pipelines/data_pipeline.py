from zenml import pipeline
from steps.data_import_step import import_and_catch_data_step
from steps.feature_extraction_preprocessing_step import feature_extraction_step, feature_preprocessing_step
from app.active_config import cfg

@pipeline
def data_pipeline():
    # Raw data importing step!
    for raw_data_dir in cfg.SETUP_RAW_DIRS:
        # Import raw data from directories!
        import_and_catch_data_step(cfg.SETUP_Name, raw_data_dir, cfg.PICKLE_DATA_DIR)

    # Feature extraction step!
    feature_extraction_step(cfg.PICKLE_DATA_DIR, cfg.FEATURE_DIR)

    # Feature Preprocessing step!
    X_train_scaled, Ene_RUL_order_train, X_test_scaled = feature_preprocessing_step(
        feature_directory=cfg.FEATURE_DIR,
        output_directory=cfg.OUTPUT_DIR,
        bearing_used=cfg.bearing_used,
        channel_used=cfg.channel
    )

    return X_train_scaled, Ene_RUL_order_train, X_test_scaled
