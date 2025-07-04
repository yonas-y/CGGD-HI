from pipelines.data_pipeline import data_pipeline

if __name__ == "__main__":
    # Run the data import, feature extract, preprocess pipeline!
    data_outputs = data_pipeline()

    # Run model training pipeline!
