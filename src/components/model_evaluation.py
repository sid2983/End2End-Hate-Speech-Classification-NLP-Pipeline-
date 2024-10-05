import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from src.components.data_transformation import DataTransformationConfig, DataTransformation
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification
from dataclasses import dataclass
from src.logger.loggings import logging
from src.exception.exception import CustomException

# Model evaluation configuration with paths to saved model and preprocessor
@dataclass
class ModelEvaluationConfig:
    model_path: str = os.path.join("artifacts", "final_model")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.evaluation_config = config
        self.model = self.load_model(self.evaluation_config.model_path)
        self.preprocessor = self.load_preprocessor(self.evaluation_config.preprocessor_path)

    def load_model(self, model_path):
        """Loads the trained model from the given path."""
        try:
            logging.info(f"Loading model from {model_path}")
            model = TFRobertaForSequenceClassification.from_pretrained(model_path)

            #compile the model again for evaluation
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
            
            # Compile the model
            model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
            logging.info("Model loaded successfully")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def load_preprocessor(self, preprocessor_path):
        """Loads the preprocessor (tokenizer and cleaning function)."""
        try:
            logging.info(f"Loading preprocessor from {preprocessor_path}")
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logging.info("Preprocessor loaded successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_text(self, texts):
        """Cleans and tokenizes input texts."""
        clean_texts = [self.preprocessor['clean_text'](text) for text in texts]
        encodings = self.preprocessor['tokenizer'](
            clean_texts, truncation=True, padding='max_length', max_length=128, return_tensors='tf'
        )
        return encodings['input_ids'], encodings['attention_mask']

    def evaluate_model_performance(self):
        """Evaluates the model on the test data."""
        try:
            logging.info("Evaluating model performance on test data...")

            # Load and preprocess test data
            test_data = pd.read_csv(self.evaluation_config.test_data_path)
            test_data = test_data.dropna(subset=['tweet', 'label'])
            test_texts = test_data['tweet'].tolist()
            test_labels = np.array([int(label) for label in test_data['label'].tolist()])

            # Tokenize test data
            input_ids, attention_masks = self.preprocess_text(test_texts)

            # Create TensorFlow dataset for testing
            test_dataset = tf.data.Dataset.from_tensor_slices(
                ({'input_ids': input_ids, 'attention_mask': attention_masks}, test_labels)
            ).batch(128)

            # Evaluate the model on test dataset
            result = self.model.evaluate(test_dataset)
            loss, accuracy = result[0], result[1]
            logging.info(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
            
            return loss, accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def classify_single_sentence(self, sentence):
        """Classifies a single sentence and returns the predicted class."""
        try:
            logging.info("Classifying a single sentence...")
            # Preprocess the sentence
            input_ids, attention_mask = self.preprocess_text([sentence])
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            # Make prediction
            predictions = self.model.predict(inputs)
            predicted_class = np.argmax(predictions.logits, axis=1)[0]
            logging.info(f"Predicted class: {predicted_class}")

            return predicted_class

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self):
        """Main function to trigger model evaluation and single sentence classification."""
        logging.info("Model Evaluation Started")
        try:
            # Perform model performance evaluation
            loss, accuracy = self.evaluate_model_performance()

            # Example of single sentence classification
            example_sentence = "I hate everything!"
            predicted_class = self.classify_single_sentence(example_sentence)
            logging.info(f"Prediction for the example sentence '{example_sentence}': {predicted_class}")

            return loss, accuracy, predicted_class

        except Exception as e:
            logging.error("Model Evaluation Failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = ModelEvaluationConfig()
    model_evaluation = ModelEvaluation(config)
    model_evaluation.initiate_model_evaluation()
