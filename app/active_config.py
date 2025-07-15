# active_config.py
import tensorflow as tf
from app.config import get_config, update_config
import os

SETUP = "XJTU_SY"         # "pronostia or XJTU_SY"
Local = False                # True=local, False=remote

if not Local:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NUMEXPR_MAX_THREADS"] = "32"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# Get the base config for the chosen setup
cfg = get_config(setup_name=SETUP, Local=Local)

# Apply dynamic update immediately
update_config(cfg,
              bearing_used='Bearing3',
              channel='both',
              n_fft=1024,
              hop_length=512,
              n_mels=128)

# ======== Change model hyperparameters when necessary! ========= #
cfg.model_hyperparams.encoding_n = 16
cfg.model_hyperparams.dropout_rate = 0.0
cfg.model_hyperparams.regularization = 1e-3
cfg.model_hyperparams.activation_function = tf.keras.layers.LeakyReLU(alpha=0.0)

# ======== Update the model training parameters when necessary! ========= #
# Change the segment partition percentage if necessary!
cfg.model_training_params.run_partitioning_portion = (0.1, 0.85, 0.05)  # first 10%, middle 85% and final 5%!!

# Change the segment percentages which makes a batch if necessary!
cfg.model_training_params.segment_percentages_in_batch = (0.2, 0.7, 0.1)

cfg.model_training_params.training_iterations = 10
cfg.model_training_params.epochs = 150
cfg.model_training_params.patience = 15
cfg.model_training_params.batch_size = 16
cfg.model_training_params.validation_split = 0.1
cfg.model_training_params.save_weights = True
cfg.model_training_params.early_stop = True

# ======== Update the rescale factors of the constraints ========= #
cfg.constraint_params.soft_rank_rf = 0.0
cfg.constraint_params.monotonicity_rf = (0.0, 0.0)
cfg.constraint_params.energy_hi_dev_rf = 0.0
cfg.constraint_params.upper_bound_rf = 0.0
cfg.constraint_params.lower_bound_rf = 0.0
