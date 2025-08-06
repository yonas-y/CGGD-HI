from zenml import step
import tensorflow as tf
from app.model_development import ConvAE_model_subclass
from app.materializers.convae_materializer import ConvAEMaterializer
from app.active_config import cfg
import logging

logger = logging.getLogger(__name__)

@step(enable_cache=False, output_materializers=ConvAEMaterializer)
def model_development_step() -> ConvAE_model_subclass:
    """
    ZenML step that builds and returns a convolutional autoencoder model
    for health indicator (HI) estimation.

    Returns:
        A compiled tf.keras.Model instance.
    """
    input_sh = cfg.model_hyperparams.input_sample_shape
    encod_n = cfg.model_hyperparams.encoding_n
    regularization = cfg.model_hyperparams.regularization
    dropout_rate = cfg.model_hyperparams.dropout_rate
    poolsize = cfg.model_hyperparams.pooling_size
    kernelsize = cfg.model_hyperparams.kernel_size_conv
    stride = cfg.model_hyperparams.stride_conv
    activation = cfg.model_hyperparams.activation_function

    logger.info(f"🧰 Initializing the model and this is its architecture!")
    model_instance = ConvAE_model_subclass(input_shape=input_sh,
                                           encoding_neurons=encod_n,
                                           reg=regularization, dropout=dropout_rate, poolsize=poolsize,
                                           kernelsize=kernelsize, stride=stride,
                                           activation=activation)
    functional_model = model_instance.model()
    functional_model.summary()

    return model_instance
