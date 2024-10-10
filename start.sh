#!/bin/sh
rm -f /app/airflow/airflow-webserver.pid && echo "Removed airflow-webserver.pid file"

python3 -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"


cd /app

#check if gpu is available

# Push DVC artifacts to S3
echo "Pushing DVC artifacts to S3..."
dvc push && echo "DVC artifacts pushed to S3 successfully"


# Start Airflow services in the background
nohup airflow scheduler &
airflow webserver 

