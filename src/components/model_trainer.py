import os
import sys
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from transformers import TFRobertaForSequenceClassification
from src.exception.exception import CustomException
from src.logger.loggings import logging
from dataclasses import dataclass

# Model trainer configuration
@dataclass
class ModelTrainerConfig:
    transformed_train_data_path: str = os.path.join("artifacts", "train_data.tfrecord")
    transformed_test_data_path: str = os.path.join("artifacts", "test_data.tfrecord")
    model_save_path: str = os.path.join("artifacts", "final_model")
    log_dir: str = './logs'
    checkpoint_path: str = './best_model'
    num_labels: int = 2
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    registered_model_name: str = "HateSpeechClassifier"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.trainer_config = config

    def load_dataset(self, path):
        try:
            dataset = tf.data.experimental.load(path)
            logging.info(f"Dataset loaded successfully from {path}")
            return dataset
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self):
        try:
            logging.info("Model Training Started")

            mlflow.set_experiment("Hate Speech Classification NLP")
            with mlflow.start_run() as run:
                mlflow.log_param("epochs", self.trainer_config.epochs)
                mlflow.log_param("batch_size", self.trainer_config.batch_size)
                mlflow.log_param("learning_rate", self.trainer_config.learning_rate)
            
                

                
                # Load the transformed datasets
                train_data = self.load_dataset(self.trainer_config.transformed_train_data_path)
                test_data = self.load_dataset(self.trainer_config.transformed_test_data_path)
                
                # Load the pre-trained Roberta model for sequence classification
                model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=self.trainer_config.num_labels)
                
                # Define optimizer, loss, and metrics
                optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
                
                # Compile the model
                model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

                # TensorBoard and Model Checkpoint callbacks
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.trainer_config.log_dir)
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.trainer_config.checkpoint_path, 
                                                                        save_best_only=True, 
                                                                        monitor='val_accuracy', 
                                                                        mode='max')
                
                # Start training the model
                history = model.fit(
                    train_data,
                    validation_data=test_data,
                    epochs=self.trainer_config.epochs,
                    callbacks=[tensorboard_callback, checkpoint_callback]
                )

                for epoch in range(self.trainer_config.epochs):
                    mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                    mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)


                # Evaluate the model on the test set
                eval_results = model.evaluate(test_data)
                loss_value, accuracy_value = eval_results[0], eval_results[1]
                logging.info(f"Evaluation results: {eval_results}")

                mlflow.log_metric("test_loss", loss_value)
                mlflow.log_metric("test_accuracy", accuracy_value)
                
                # Save the final model
                model.save_pretrained(self.trainer_config.model_save_path)
                logging.info("Model saved successfully")

                mlflow.tensorflow.log_model(tf_saved_model_dir=self.trainer_config.model_save_path, 
                                            artifact_path="model", 
                                            registered_model_name=self.trainer_config.registered_model_name)
                
                mlflow.log_artifacts(self.trainer_config.log_dir, artifact_path="tensorboard_logs")

                logging.info("Model Training Completed")

                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri=model_uri, name=self.trainer_config.registered_model_name)
                logging.info(f"Model registered under name: {self.trainer_config.registered_model_name}")
                return eval_results

        except Exception as e:
            logging.error("Model Training Failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = ModelTrainerConfig()
    trainer = ModelTrainer(config)
    trainer.initiate_model_trainer()