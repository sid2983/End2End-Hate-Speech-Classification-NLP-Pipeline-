from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import timedelta
from zoneinfo import ZoneInfo

# Define the DAG with appropriate configurations
with DAG(
    "robust_flexible_nlp_pipeline",
    default_args={
        "retries": 2,
    },
    description="DAG for training NLP model",
    schedule=timedelta(minutes=60),
    start_date=pendulum.datetime(2024, 10, 10, tz=ZoneInfo('Asia/Kolkata')),
    concurrency=1,
    catchup=False,
    tags=["machine_learning", "NLP", "Hate_Speech"],
) as dag:
    
    dag.doc_md = __doc__

    # Task for Data Ingestion
    data_ingestion_task = BashOperator(
        task_id="data_ingestion",
        bash_command="cd /app && dvc repro -s data_ingestion --force >> logs/data_ingestion.log 2>&1",
    )

    data_ingestion_task.doc_md = dedent(
        """
        #### Data Ingestion Task
        This task is responsible for ingesting the data.
        """
    )

    # Task for Data Transformation
    data_transformation_task = BashOperator(
        task_id="data_transformation",
        bash_command=" cd /app && dvc repro -s data_transformation --force >> logs/data_transformation.log 2>&1",
    )

    data_transformation_task.doc_md = dedent(
        """
        #### Data Transformation Task
        This task is responsible for transforming the data.
        """
    )

    # Task for Model Training
    model_trainer_task = BashOperator(
        task_id="model_trainer",
        bash_command=" cd /app && dvc repro -s model_training --force >> logs/model_training.log 2>&1",
    )

    model_trainer_task.doc_md = dedent(
        """
        #### Model Training Task
        This task is responsible for SFT of  the BERT model.
        """
    )

    # Task for Pushing Artifacts to S3
    push_to_s3_task = BashOperator(
        task_id="push_to_s3",
        bash_command="cd /app && dvc push >> /app/logs/dvc_push.log 2>&1",
    )

    # Define the execution flow
    data_ingestion_task >> data_transformation_task >> model_trainer_task >> push_to_s3_task
