from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='train_pipeline',
    default_args=default_args,
    description='Pipeline for training model',
    schedule_interval=None,
    catchup=False,
) as dag:

    # Шаг 1: Генерация данных
    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command='cd /opt/airflow && python src/data/make_dataset.py',
    )

    # Шаг 2: Обучение
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /opt/airflow && python src/models/train_model.py',
    )

    prepare_data >> train_model