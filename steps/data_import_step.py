from zenml import step
import os
import logging

from app.data_importing import import_bearing_data_to_pickle

logger = logging.getLogger(__name__)

@step
def import_and_catch_data_step(raw_data_dir, pickle_dir) -> None:
    """
    Step to load raw CSVs, combine and save as pickle, only if pickle doesn't already exist.
    """
    if os.path.exists(pickle_dir):
        logger.info(f"✅ Pickle already exists at {pickle_dir}, skipping import.")
    else:
        logger.info(f"📦 Pickle not found at {pickle_dir}. Importing raw data...")
        import_bearing_data_to_pickle(raw_data_dir, pickle_dir)

    return
