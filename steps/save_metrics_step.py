from zenml import step
import os
import pandas as pd

from app.active_config import cfg

import logging
logger = logging.getLogger(__name__)

@step(enable_cache=False)
def save_metrics_step(performance_df: pd.DataFrame, iteration_n: int):
    output_dir = 'output/model_performance'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'performance_df_' \
               f'{cfg.model_hyperparams.encoding_n}_' \
               f'{cfg.SETUP_Name}_' \
               f'{cfg.constraint_params.reconstruction_rf}_' \
               f'{cfg.constraint_params.soft_rank_rf}_' \
               f'{cfg.constraint_params.monotonicity_rf[1]}_' \
               f'{cfg.constraint_params.energy_hi_dev_rf}_' \
               f'{cfg.constraint_params.upper_bound_rf}_' \
               f'{cfg.constraint_params.lower_bound_rf}_' \
               f'{iteration_n}.pkl'

    performance_df.to_pickle(os.path.join(output_dir, filename))
    logger.info(f"✅  Saved model performance as: {filename}")
