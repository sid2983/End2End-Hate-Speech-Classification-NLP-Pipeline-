import os
import sys
import re
import emoji
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception.exception import CustomException
from src.logger.loggings import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

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

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            # Load the data from CSV
            data = pd.read_csv("https://raw.githubusercontent.com/sid2983/datahub/refs/heads/master/hate_speech_data.csv")  # Replace with actual data source path
            logging.info("Data Loaded Successfully")

            # Apply the clean_text method to clean the text data
            logging.info("Cleaning the data using the clean_text method")
            data['tweet'] = data['tweet'].apply(self.clean_text)  # Replace 'text_column' with the actual text column name

            # Save the raw cleaned data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Cleaned data saved successfully in the Artifacts folder")

            # Perform train-test split
            logging.info("Performing train-test split")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info("Train-test split completed")

            # Save train and test datasets
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test data saved successfully in the Artifacts folder")
            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Data Ingestion Failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    data_ingestion.initiate_data_ingestion()
    