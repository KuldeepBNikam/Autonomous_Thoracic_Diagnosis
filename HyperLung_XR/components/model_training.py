import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR,_LRScheduler
from tqdm import tqdm
from torch.optim import Optimizer
from torch.nn import Module
import bentoml
import joblib

from HyperLung_XR.entity.config_entity import ModelTrainerConfig
from HyperLung_XR.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from HyperLung_XR.exception import HyperLungException
from HyperLung_XR.logger import logging
from HyperLung_XR.ml.model.arch import HybridCNNTransformer
from HyperLung_XR.constant.training_pipeline import *


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config
        self.data_transformation_artifact: DataTransformationArtifact = data_transformation_artifact
        self.device = DEVICE

        self.model: HybridCNNTransformer = HybridCNNTransformer(num_classes=2).to(self.device)

    def train(self, optimizer:Optimizer):
        """
        Description: To train the model

        input: model,device,train_loader,optimizer,epoch

        output: loss, batch id and accuracy
        """
        logging.info("Entered the train method of Model trainer class")
        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct: int = 0

            processed = 0

            class_weights = torch.tensor([1.0, 1.5]).to(self.device)


            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Initialization of gradient
                optimizer.zero_grad()

                # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                # or specific requirements
                ## prediction on data

                y_pred = self.model(data)

                # Calculating loss given the prediction
                loss = F.cross_entropy(y_pred, target, weight=class_weights)


                # Backprop
                loss.backward()

                optimizer.step()

                # get the index of the log-probability corresponding to the max value
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100*correct/processed:.2f}%")
                
            self.last_train_accuracy = 100 * correct / processed


            logging.info("Exited the train method of Model trainer class")

        except Exception as e:
            raise HyperLungException(e, sys)
    

    def validate(self) -> float:
        logging.info("Entered validation method")

        try:
            self.model.eval()
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for data, target in self.data_transformation_artifact.transformed_val_object:
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    output = self.model(data)
                    val_loss += F.cross_entropy(
                        output, target, reduction="sum"
                    ).item()

                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

            val_loss /= len(
                self.data_transformation_artifact.transformed_val_object.dataset
            )

            val_accuracy = 100.0 * correct / len(
                self.data_transformation_artifact.transformed_val_object.dataset
            )

            logging.info(
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.2f}%"
            )

            return val_accuracy

        except Exception as e:
            raise HyperLungException(e, sys)

        

    def test(self) -> None:
        try:
            """
            Description: To test the model

            input: model, DEVICE, test_loader

            output: average loss and accuracy

            """
            logging.info("Entered the test method of Model trainer class")

            self.model.eval()

            test_loss: float = 0.0

            correct: int = 0

            with torch.no_grad():
                for (
                    data,
                    target,
                ) in self.data_transformation_artifact.transformed_test_object:
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    output = self.model(data)

                    test_loss += F.cross_entropy(output, target, reduction="sum").item()

                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(
                    self.data_transformation_artifact.transformed_test_object.dataset
                )

                print(
                    "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        100.0
                        * correct
                        / len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                    )
                )

            logging.info(
                f"🧪 FINAL TEST RESULTS | "
                f"Loss: {test_loss:.4f} | "
                f"Accuracy: {100.0 * correct / len(self.data_transformation_artifact.transformed_test_object.dataset):.2f}%"
            )


            logging.info("Exited the test method of Model trainer class")

        except Exception as e:
            raise HyperLungException(e, sys)
        

        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            model: Module = self.model.to(self.model_trainer_config.device)

            optimizer = AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)


            scheduler: _LRScheduler = StepLR(
                optimizer=optimizer, **self.model_trainer_config.scheduler_params
            )
            best_val_accuracy = 0.0
            patience = 3
            patience_counter = 0


            fine_tuning_started = False


            for epoch in range(1, self.model_trainer_config.epochs + 1):
                print(f"\nEpoch {epoch}")

                # 🔓 Start fine-tuning at epoch 7
                if epoch == 7 and not fine_tuning_started:
                    print("🔓 Unfreezing top layers for fine-tuning")

                    self.model.unfreeze_top_layers()

                    optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=1e-5,
                        weight_decay=1e-4
                    )
                    scheduler = StepLR(
                    optimizer=optimizer,
                    **self.model_trainer_config.scheduler_params)


                    fine_tuning_started = True

                self.train(optimizer=optimizer)
                val_accuracy = self.validate()

                logging.info(
                    f"[EPOCH {epoch}] "
                    f"Train Acc: {self.last_train_accuracy:.2f}% | "
                    f"Val Acc: {val_accuracy:.2f}% | "
                    f"Best Val: {best_val_accuracy:.2f}%"
                )


                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0

                    os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)
                    torch.save(model.state_dict(), self.model_trainer_config.trained_model_path)
                    logging.info(
                        f"📌 Model improved → saving checkpoint "
                        f"(Val Acc: {val_accuracy:.2f}%)"
                    )


                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.warning(
                        f"🛑 Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )

                    break

                scheduler.step()

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            

            train_transforms_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model(
                name=self.model_trainer_config.trained_bentoml_model_name,
                model=model,
                custom_objects={
                    self.model_trainer_config.train_transforms_key: train_transforms_obj
                },
                weights_only=True   
            )

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path,
                class_mapping=PREDICTION_LABEL,
                device=str(self.device)
            )

            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise HyperLungException(e, sys)