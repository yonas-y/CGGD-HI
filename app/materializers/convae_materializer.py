import os
import tensorflow as tf
from typing import Type
from app.model_development import ConvAE_model_subclass
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.logger import get_logger

logger = get_logger(__name__)

class ConvAEMaterializer(BaseMaterializer):
    """Simple materializer to save/load ConvAE_model_subclass models."""

    ASSOCIATED_TYPES = (ConvAE_model_subclass, tf.keras.Model)

    def save(self, model: ConvAE_model_subclass) -> None:
        """Save the model to the artifact store."""
        model_path = os.path.join(self.uri, "model.keras")
        logger.info(f"Saving ConvAE model to: {model_path}")
        model.save(model_path)

    def load(self, data_type: Type[ConvAE_model_subclass]) -> ConvAE_model_subclass:
        """Load the model from the artifact store."""
        model_path = os.path.join(self.uri, "saved_model_dir")
        logger.info(f"Loading ConvAE model from: {model_path}")
        functional_model = tf.keras.models.load_model(model_path, compile=False)
        # ⚠️ Return functional model or wrap again if needed
        return functional_model
