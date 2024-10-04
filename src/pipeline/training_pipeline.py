import os
import sys
import argparse

from src.exception.exception import CustomException
from src.logger.loggings import logging
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(DataIngestionConfig())
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            return train_data_path, test_data_path   
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self):
        try:
            data_transformation = DataTransformation(DataTransformationConfig())
            transformed_train_data_path, transformed_test_data_path = data_transformation.initiate_data_transformation()
            return transformed_train_data_path, transformed_test_data_path
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_training(self):
        try:
            # Instantiate the ModelTrainerConfig without modifying it
            model_trainer_config = ModelTrainerConfig()
            model_trainer = ModelTrainer(model_trainer_config)
            model_trainer.initiate_model_trainer()  # This will load datasets from the paths in the config
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_training_pipeline(self):
        try:
            self.start_data_ingestion()
            self.start_data_transformation()
            self.start_model_training()

        except Exception as e:
            raise CustomException(e, sys)

def main():
    parser = argparse.ArgumentParser(description="Run stages of the training pipeline.")
    parser.add_argument('stage', choices=['data_ingestion', 'data_transformation', 'model_training', 'full_pipeline'],
                        help="Specify the stage of the pipeline to run.")
    args = parser.parse_args()

    pipeline = TrainingPipeline()
    

    if args.stage == 'data_ingestion':
        logging.info(f"Running pipeline stage: {args.stage}")
        pipeline.start_data_ingestion()

    elif args.stage == 'data_transformation':
        logging.info(f"Running pipeline stage: {args.stage}")
        pipeline.start_data_transformation()

    elif args.stage == 'model_training':
        logging.info(f"Running pipeline stage: {args.stage}")
        pipeline.start_model_training()  # Just call it without arguments

    elif args.stage == 'full_pipeline':
        logging.info(f"Running pipeline stage: {args.stage}")
        pipeline.start_training_pipeline()

if __name__ == '__main__':
    main()
