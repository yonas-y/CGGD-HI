from zenml import step
import pandas as pd
from typing import List
from app.model_execution import model_execution

import logging

logger = logging.getLogger(__name__)

@step(enable_cache=False)
def model_execution_step(model,
                         training_data: List,
                         validation_data: List,
                         iteration_n: int) -> pd.DataFrame:
    """
    ZenML step that executed the model using the COnvAE architecture
    for health indicator (HI) estimation.

    Returns:
        A pandas data frame.
    """
    logger.info(f"🧰 Initializing the model execution phase!")
    model_performance_df = model_execution(model=model,
                                           training_data=training_data,
                                           validation_data=validation_data,
                                           iteration=iteration_n)

    return model_performance_df
