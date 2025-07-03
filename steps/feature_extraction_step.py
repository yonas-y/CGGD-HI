from zenml import step
import os
import logging

from app.feature_extraction import feature_extraction

logger = logging.getLogger(__name__)

@step(enable_cache=True)
def feature_extraction_step(pickle_dir, feature_dir) -> None:
    """
    Step to extract features from the input pickle data.
    """
    logger.info(f"📦 Extracting features ...")
    feature_extraction(pickle_dir, feature_dir)

    return
