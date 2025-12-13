import sys

from HyperLung_XR.cloud_storage.s3_operation import S3operation
#from HyperLung_XR.constant.training_pipeline import *
from HyperLung_XR.entity.artifact_entity import DataIngestionArtifact
from HyperLung_XR.entity.config_entity import DataIngestionConfig
from HyperLung_XR.exception import HyperLungException
from HyperLung_XR.logger import loggging

class DataIngestion:
    def __init__(self):
        pass

    def get_data_from_s3(self):
        try:
            pass

        except Exception as e:
            raise HyperLungException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            pass

        except Exception as e:
            raise HyperLungException(e, sys)