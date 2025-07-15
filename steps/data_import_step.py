from zenml import step
import os
import logging

from app.data_importing import import_pronostia_data_to_pickle, import_XJTU_SY_data_to_pickle

logger = logging.getLogger(__name__)

@step(enable_cache=False)
def import_and_catch_data_step(setup_name, raw_data_dir, pickle_dir) -> bool:
    """
    Step to load raw CSVs, combine and save as pickle.
    """
    if setup_name == "pronostia":
        logger.info("📦 Importing pronostia raw data as a pickle file...")
        import_pronostia_data_to_pickle(raw_data_dir, pickle_dir)
    elif setup_name == "XJTU_SY":
        logger.info("📦 Importing XJTU_SY raw data as a pickle file...")
        import_XJTU_SY_data_to_pickle(raw_data_dir, pickle_dir)

    return True
