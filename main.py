import sys

from HyperLung_XR.exception import HyperLungException
from HyperLung_XR.pipeline.training_pipeline import TrainPipeline

def start_training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

    except Exception as e:
        raise HyperLungException(e,sys)
    

if __name__ == "__main__":
    start_training()