# active_config.py
import tensorflow as tf
from app.config import get_config, update_config

SETUP = "pronostia"        # "pronostia or XJTU_SY"
# Get the base config for the chosen setup
cfg = get_config(SETUP)

# Apply dynamic update immediately
update_config(cfg,
              bearing_used='Bearing3',
              channel='both',
              n_fft=1024,
              hop_length=512,
              n_mels=128)

# ======== Change model hyperparameters when necessary! ========= #
cfg.model_hyperparams.encoding_n = 64
cfg.model_hyperparams.dropout_rate = 0.0
cfg.model_hyperparams.regularization = 1e-3
cfg.model_hyperparams.activation_function = tf.keras.layers.LeakyReLU(negative_slope=0.0)


# ======== Update the model training parameters when necessary! ========= #
# Change the segment partition percentage if necessary!
cfg.model_training_params.run_partitioning_portion = (0.1, 0.85, 0.05)  # first 10%, middle 85% and final 5%!!

# Change the segment percentages which makes a batch if necessary!
cfg.model_training_params.segment_percentages_in_batch = (0.2, 0.7, 0.1)

cfg.model_training_params.batch_size = 64
cfg.model_training_params.validation_split = 0.2
cfg.model_training_params.epochs = 100
cfg.model_training_params.patience = 15
cfg.model_training_params.training_iterations = 10
cfg.model_training_params.save_weights = False
cfg.model_training_params.early_stop = False


# ======== Update the rescale factors of the constraints ========= #
cfg.constraint_params.soft_rank_rf = 1.0
cfg.constraint_params.monotonicity_rf = (1.25, 1.5)
cfg.constraint_params.energy_hi_dev_rf=1.5
cfg.constraint_params.upper_bound_rf=2.0
cfg.constraint_params.lower_bound_rf=2.0

