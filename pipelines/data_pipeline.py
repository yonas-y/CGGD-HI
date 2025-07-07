from zenml import pipeline
from steps.data_import_step import import_and_catch_data_step
from steps.feature_extraction_preprocessing_step import feature_extraction_step, feature_preprocessing_step
from app.active_config import cfg

@pipeline
def data_pipeline():
    # Raw data importing step!
    import_and_catch_data_step(cfg.TRAIN_RAW_DATA_DIR, cfg.PICKLE_TRAIN_DIR)   # Import the training set!
    import_and_catch_data_step(cfg.TEST_RAW_DATA_DIR, cfg.PICKLE_TEST_DIR)    # Import the test set!

    # Feature extraction step!
    feature_extraction_step(cfg.PICKLE_TRAIN_DIR, cfg.FEATURE_DIR) # Feature extraction for the training set!
    feature_extraction_step(cfg.PICKLE_TEST_DIR, cfg.FEATURE_DIR) # Feature extraction for the test set!

    # Feature Preprocessing step!
    X_train_scaled, Ene_RUL_order_train, X_test_scaled = feature_preprocessing_step(
        feature_directory=cfg.FEATURE_DIR,
        output_directory=cfg.OUTPUT_DIR,
        bearing_used=cfg.bearing_used,
        channel_used=cfg.channel
    )

    return X_train_scaled, Ene_RUL_order_train, X_test_scaled
