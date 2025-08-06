from zenml import pipeline
from steps.model_development_step import model_development_step
@pipeline
def model_development_pipeline():

    # Model architecture development step!
    model = model_development_step()

    return model
