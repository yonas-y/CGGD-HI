from zenml import pipeline
from steps.data_import_step import import_and_catch_data_step
from steps.feature_extraction_step import feature_extraction_step
from app.config import TRAIN_RAW_DATA_DIR, TEST_RAW_DATA_DIR, PICKLE_TRAIN_DIR, PICKLE_TEST_DIR, FEATURE_DIR

@pipeline
def data_pipeline():
    # Raw data importing step!
    import_and_catch_data_step(TRAIN_RAW_DATA_DIR, PICKLE_TRAIN_DIR)   # Import the training set!
    import_and_catch_data_step(TEST_RAW_DATA_DIR, PICKLE_TEST_DIR)    # Import the test set!

    # Feature extraction step!
    feature_extraction_step(PICKLE_TRAIN_DIR, FEATURE_DIR) # Feature extraction for the training set!
    feature_extraction_step(PICKLE_TEST_DIR, FEATURE_DIR) # Feature extraction for the test set!


