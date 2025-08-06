from zenml.client import Client
from pipelines.training_pipeline import training_pipeline
from pipelines.model_development_pipeline import model_development_pipeline

from app.active_config import cfg

import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # -----------------------------------------------
    # ✅ Step 1: Building the model architecture used!
    # -----------------------------------------------
    model_run = model_development_pipeline()
    client = Client()

    run = client.get_pipeline_run(str(model_run.id))
    step_run = run.steps["model_development_step"]
    convAE_model_artifact = step_run.outputs["output"][0]

    # -----------------------------------------------
    # ✅ Step 2: Load the data, train and validate!
    # -----------------------------------------------
    iteration_range = cfg.model_training_params.training_iterations
    for iter_n in range(iteration_range[0], iteration_range[1]):
        logger.info(f"🚀 Started model execution step {iter_n + 1} ...... ")
        training_pipeline(model=convAE_model_artifact, iter_n=iter_n)
