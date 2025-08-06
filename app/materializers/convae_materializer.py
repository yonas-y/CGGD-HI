import os
import tensorflow as tf
from typing import Type
from app.model_development import ConvAE_model_subclass
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.logger import get_logger

logger = get_logger(__name__)

class ConvAEMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (ConvAE_model_subclass,)

    def save(self, model: ConvAE_model_subclass) -> None:
        path = os.path.join(self.uri, "model.keras")
        model.model().save(path)

    def load(self, data_type: Type[ConvAE_model_subclass]) -> ConvAE_model_subclass:
        path = os.path.join(self.uri, "model.keras")
        functional_model = tf.keras.models.load_model(path, compile=False)
        # re-wrap into ConvAE_model_subclass if needed
        return functional_model
