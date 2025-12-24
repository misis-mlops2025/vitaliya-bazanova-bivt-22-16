from datetime import datetime
from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.bash import BashOperator

FILE_PATH = '/opt/data/new_data.csv'

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

with DAG(
    dag_id='inference_pipeline',
    default_args=default_args,
    description='Pipeline for inference',
    schedule_interval=None,
    catchup=False,
) as dag:

    # 1. Ждем файл
    wait_for_file = FileSensor(
        task_id='wait_for_file',
        filepath=FILE_PATH,
        poke_interval=5,
        timeout=600,
        mode='poke'
    )

    # 2. Делаем предсказание
    predict = BashOperator(
        task_id='predict',
        bash_command=f'cd /opt/airflow && python src/models/predict_model.py {FILE_PATH}',
    )

    # 3. Удаляем исходный файл
    remove_file = BashOperator(
        task_id='remove_file',
        bash_command=f'rm {FILE_PATH}',
    )

    wait_for_file >> predict >> remove_file