from zenml import pipeline
from steps.data_import_step import import_and_catch_data
from app.config import TRAIN_RAW_DATA_DIR, TEST_RAW_DATA_DIR, PICKLE_TRAIN_DIR, PICKLE_TEST_DIR
@pipeline
def data_pipeline():
    import_and_catch_data(TRAIN_RAW_DATA_DIR, PICKLE_TRAIN_DIR)   # Import the training set!
    import_and_catch_data(TEST_RAW_DATA_DIR, PICKLE_TEST_DIR)    # Import the test set!
