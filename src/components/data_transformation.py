import os
import sys
import re
import emoji
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
from transformers import RobertaTokenizer
from dataclasses import dataclass
from src.logger.loggings import logging
from src.exception.exception import CustomException

# Data transformation configuration with artifact paths
@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    transformed_train_data_path: str = os.path.join("artifacts", "train_data.tfrecord")
    transformed_test_data_path: str = os.path.join("artifacts", "test_data.tfrecord")
    preprocessor_save_path: str = os.path.join("artifacts", "preprocessor.pkl")
    tokenizer_path: str = 'roberta-base'
    max_seq_length: int = 128

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.transformation_config = config
        self.tokenizer = RobertaTokenizer.from_pretrained(self.transformation_config.tokenizer_path)

    def read_data(self):
        try:
            train_data = pd.read_csv(self.transformation_config.train_data_path)
            test_data = pd.read_csv(self.transformation_config.test_data_path)
            return train_data, test_data
        except Exception as e:
            raise CustomException(e, sys)
        
    def clean_text(self, text: str) -> str:


        text = text.lower()

        text = emoji.demojize(text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s,\.!?]', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)  
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def clean_and_tokenize_data(self, texts, labels):
        # Tokenization process with truncation, padding, and max length
        logging.info(" Cleaning Tokenizing data...")

        texts = [self.clean_text(text) for text in texts]

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Input contains non-string data")
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.transformation_config.max_seq_length, return_tensors='tf')
        return encodings['input_ids'], encodings['attention_mask'], np.array(labels)

    def create_tf_dataset(self, input_ids, attention_masks, labels):
        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))
        return dataset
    
    def save_preprocessor(self):
        """Saves the preprocessor, including the tokenizer and cleaning logic."""
        try:
            preprocessor = {
                'tokenizer': self.tokenizer,
                'clean_text': self.clean_text
            }
            with open(self.transformation_config.preprocessor_save_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            logging.info(f"Preprocessor saved to {self.transformation_config.preprocessor_save_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info("Data Transformation Started")
        try:
            # Reading data
            train_df, test_df = self.read_data()

            train_df = train_df.dropna(subset=['tweet', 'label'])
            test_df = test_df.dropna(subset=['tweet', 'label'])

            train_df['tweet'] = train_df['tweet'].astype(str)
            test_df['tweet'] = test_df['tweet'].astype(str)
            
            # Cleaning and Tokenizing
            train_texts = train_df['tweet'].tolist()
            train_labels = train_df['label'].tolist()
            test_texts = test_df['tweet'].tolist()
            test_labels = test_df['label'].tolist()

            # Check if any label is not an integer
            try:
                if not all(isinstance(label, int) for label in train_labels):
                    raise ValueError("Labels are not integers")
            except Exception as e:
                print(e)
                train_labels = [int(label) for label in train_labels]
                test_labels = [int(label) for label in test_labels]

            train_input_ids, train_attention_masks, train_labels = self.clean_and_tokenize_data(train_texts, train_labels)
            test_input_ids, test_attention_masks, test_labels = self.clean_and_tokenize_data(test_texts, test_labels)

            # Create TF datasets
            train_dataset = self.create_tf_dataset(train_input_ids, train_attention_masks, train_labels)
            test_dataset = self.create_tf_dataset(test_input_ids, test_attention_masks, test_labels)

            # Batching and Prefetching
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            train_data = (train_dataset
                        .shuffle(buffer_size=1000)  # Shuffle the dataset
                        .batch(64)  # Batch size for training
                        .prefetch(buffer_size=AUTOTUNE))  # Prefetch for efficient training

            test_data = (test_dataset
                        .batch(128)  # Batch size for validation/testing
                        .prefetch(buffer_size=AUTOTUNE))  # Prefetch for efficient evaluation

            # Save the datasets using the recommended method
            logging.info("Saving transformed data as TFRecord files...")
            train_data.save(self.transformation_config.transformed_train_data_path)
            test_data.save(self.transformation_config.transformed_test_data_path)
            logging.info("Data Transformation Completed and saved to artifacts.")

            self.save_preprocessor()

            return (
                self.transformation_config.transformed_train_data_path,
                self.transformation_config.transformed_test_data_path
            )
        
        except Exception as e:
            logging.error("Data Transformation Failed")
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = DataTransformationConfig()
    data_transformation = DataTransformation(config)
    data_transformation.initiate_data_transformation()
