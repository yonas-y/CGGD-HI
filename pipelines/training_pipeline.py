from zenml import pipeline
from pipelines.data_pipeline import data_pipeline

from steps.training_step import model_execution_step
from steps.save_metrics_step import save_metrics_step

@pipeline
def training_pipeline(model, iter_n: int = 0):
    # Get data from data_pipeline!
    training_data, validation_data, _ = data_pipeline()

    # Model execution step!
    perf_df = model_execution_step(
        model=model,
        training_data=training_data,
        validation_data=validation_data,
        iteration_n=iter_n
    )

    # Save the model performance!
    save_metrics_step(performance_df=perf_df, iteration_n=iter_n)
