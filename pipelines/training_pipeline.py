from zenml import pipeline
from steps.training_steps import model_development_step

from app.active_config import cfg

@pipeline
def training_pipeline():
    # Model development step!
    convAE_model = model_development_step()

    # # Model training step!
    # feature_extraction_step(cfg.PICKLE_DATA_DIR, cfg.FEATURE_DIR)



    return
