import os
import sys

from HyperLung_XR.entity.artifact_entity import ModelPusherArtifact,ModelTrainerArtifact
from HyperLung_XR.entity.config_entity import ModelPusherConfig
from HyperLung_XR.exception import HyperLungException
from HyperLung_XR.logger import logging


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig,
                 model_trainer_artifact: ModelTrainerArtifact):
        
        self.model_pusher_config = model_pusher_config
        self.model_trainer_artifact = model_trainer_artifact


    def build_and_push_bento_image(self):
        logging.info("Entered build_and_push_bento_image method of ModelPusher class")

        try:
            logging.info("Building the bento from bentofile.yaml")

            os.system("bentoml build -f bentofile.yaml")

            logging.info("Built the bento from bentofile.yaml")

            logging.info("Creating docker image for bento")

            os.system(
            f"bentoml containerize "
            f"{self.model_pusher_config.bentoml_service_name}:latest "
            f"-t {self.model_pusher_config.bentoml_ecr_image}:latest"
        )

            logging.info("Created docker image for bento")

            logging.info("Logging into ECR")

            os.system(
                f"aws ecr get-login-password --region {self.model_pusher_config.aws_region} "
                f"| docker login --username AWS --password-stdin "
                f"{self.model_pusher_config.aws_account_id}.dkr.ecr."
                f"{self.model_pusher_config.aws_region}.amazonaws.com"
            )


            logging.info("Logged into ECR")

            logging.info("Pushing bento image to ECR")

            os.system(
                    f"docker push {self.model_pusher_config.bentoml_ecr_image}:latest"
                )


            logging.info("Pushed bento image to ECR")

            logging.info(
                "Exited build_and_push_bento_image method of ModelPusher class"
            )

        except Exception as e:
            raise HyperLungException(e, sys)
        


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            if self.model_trainer_artifact is None:
                raise Exception("ModelTrainerArtifact missing. Aborting model push.")

            self.build_and_push_bento_image()

            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name,
                bentoml_service_name=self.model_pusher_config.bentoml_service_name,
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise HyperLungException(e, sys)