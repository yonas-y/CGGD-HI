from zenml import step
import tensorflow as tf
import logging
import numpy as np
from typing import List, Tuple

from app.model_development import ConvAE_model_subclass

from app.active_config import cfg

logger = logging.getLogger(__name__)

@step(enable_cache=False)
def model_development_step() -> None:
    """
    Step to extract features from the input pickle data.
    """
    input_sh = cfg.model_hyperparams.input_sample_shape
    encod_n = cfg.model_hyperparams.encoding_n
    regularization = cfg.model_hyperparams.regularization
    dropout_rate = cfg.model_hyperparams.dropout_rate
    poolsize = cfg.model_hyperparams.pooling_size
    kernelsize = cfg.model_hyperparams.kernel_size_conv
    stride = cfg.model_hyperparams.stride_conv
    activation = cfg.model_hyperparams.activation_function

    logger.info(f"🧰 Initializing the model and this it its architecture!")
    model_instance = ConvAE_model_subclass(input_shape=input_sh,
                                           encoding_neurons=encod_n,
                                           reg=regularization, dropout=dropout_rate, poolsize=poolsize,
                                           kernelsize=kernelsize, stride=stride,
                                           activation=activation)
    functional_model = model_instance.model()
    functional_model.summary()
    return