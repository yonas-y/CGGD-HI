from zenml import step
import os
import logging

from app.data_importing import import_bearing_data_to_pickle

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def import_and_catch_data_step(raw_data_dir, pickle_dir) -> None:
    """
    Step to load raw CSVs, combine and save as pickle.
    """
    logger.info("📦 Importing raw data as a pickle file...")
    import_bearing_data_to_pickle(raw_data_dir, pickle_dir)

    return
