import os
from dataclasses import dataclass
from torch import device
from HyperLung_XR.constant.training_pipeline import *



class DataIngestionConfig:
    
    def __init__(self):

        self.s3_data_folder: str = S3_DATA_FOLDER

        self.bucket_name: str = BUCKET_NAME

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        self.data_path: str = os.path.join(self.artifact_dir,"data_ingestion",self.s3_data_folder)

        self.train_data_path: str = os.path.join(self.data_path, "train")
        
        self.val_data_path: str = os.path.join(self.data_path, "val")

        self.test_data_path: str = os.path.join(self.data_path, "test")


class DataTransformationConfig:

    def __init__(self):

        self.color_jitter_transform: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE,
            "num_workers": 2, 
            "pin_memory": PIN_MEMORY,
        }
        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )
        self.val_transforms_file: str = os.path.join(
            self.artifact_dir,  VAL_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )

class ModelTrainerConfig:
    def __init__(self):
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training")

        self.trained_bentoml_model_name: str = "xray_model"

        self.trained_model_path: str = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        self.optimizer_params: dict = {"lr": 0.01, "momentum": 0.8}

        self.scheduler_params: dict = {"step_size": STEP_SIZE, "gamma": GAMMA}

        self.device: device = DEVICE