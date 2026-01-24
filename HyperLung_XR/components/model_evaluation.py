import sys
from typing import Tuple

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from HyperLung_XR.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from HyperLung_XR.entity.config_entity import ModelEvaluationConfig
from HyperLung_XR.exception import HyperLungException
from HyperLung_XR.logger import logging
from HyperLung_XR.ml.model.arch import HybridCNNTransformer



class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):

        self.data_transformation_artifact = data_transformation_artifact

        self.model_evaluation_config = model_evaluation_config

        self.model_trainer_artifact = model_trainer_artifact

    def configuration(self) -> Tuple[DataLoader, Module, float, Optimizer]:
        logging.info("Entered the configuration method of Model evaluation class")

        try:
            test_dataloader: DataLoader = (
                self.data_transformation_artifact.transformed_test_object
            )

            model = HybridCNNTransformer(num_classes=2)

            model.load_state_dict(
                torch.load(
                    self.model_trainer_artifact.trained_model_path,
                    map_location=self.model_evaluation_config.device
                )
            )

            model.to(self.model_evaluation_config.device)
            model.eval()

            cost: Module = CrossEntropyLoss()

            logging.info("Exited the configuration method of Model evaluation class")

            return test_dataloader, model, cost

        except Exception as e:
            raise HyperLungException(e, sys)

    def test_net(self) -> float:
        logging.info("Entered the test_net method of Model evaluation class")

        try:
            test_dataloader, net, cost = self.configuration()
            total_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                
                for _, data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evaluation_config.device)

                    labels = data[1].to(self.model_evaluation_config.device)

                    output = net(images)

                    loss = cost(output, labels)

                    probs = torch.softmax(output, dim=1)
                    predictions = torch.argmax(probs, dim=1)

                    
                    total_loss += loss.item()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            avg_loss = total_loss / len(test_dataloader)
            accuracy = (correct / total) * 100

            logging.info(f"Model Evaluation → Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            logging.info("Exited the test_net method of Model evaluation class")

            return accuracy, avg_loss


        except Exception as e:
            raise HyperLungException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info(
            "Entered the initiate_model_evaluation method of Model evaluation class"
        )

        try:
            accuracy, loss = self.test_net()

            model_evaluation_artifact = ModelEvaluationArtifact(
                model_accuracy=accuracy,
                model_loss=loss
            )


            logging.info(
                "Exited the initiate_model_evaluation method of Model evaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise HyperLungException(e, sys)