from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
from feast import FeatureStore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

REPO_PATH = r"C:/Users/vitaliya.bazanova/Desktop/mlops4cours/hw1/my_feature_repo/feature_repo"

def train_model_logic():
    print(f"--- Подключаемся к Feast по пути: {REPO_PATH} ---")

    # 1. Инициализация (указываем явный путь)
    store = FeatureStore(repo_path=REPO_PATH)

    # 2. Формирование сущностей (Entity DataFrame)
    print("--- Генерация списка драйверов и дат ---")
    driver_ids = [1001, 1002, 1003, 1004, 1005]
    timestamps = [datetime.now() - timedelta(days=i) for i in range(10)]

    entity_rows = []
    for d_id in driver_ids:
        for ts in timestamps:
            entity_rows.append({
                "driver_id": d_id,
                "event_timestamp": ts
            })
    entity_df = pd.DataFrame.from_dict(entity_rows)

    # 3. Запрос фичей
    print("--- Запрос исторических фичей ---")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips"
        ]
    ).to_df()

    training_df.dropna(inplace=True)
    print(f"Размер датасета: {len(training_df)}")

    # 4. Обучение
    print("--- Обучение модели ---")
    target = "avg_daily_trips"
    features = ["conv_rate", "acc_rate"]

    X = training_df[features]
    y = training_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"SUCCESS! Model trained via Airflow. MSE: {mse}")


# --- ОПРЕДЕЛЕНИЕ DAG ---
default_args = {
    'owner': 'vitaliya',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
        dag_id='hw6_feast_training',
        default_args=default_args,
        schedule_interval=None,  # Запуск только вручную
        catchup=False,
        tags=['mlops', 'hw6']
) as dag:
    training_task = PythonOperator(
        task_id='train_feast_model',
        python_callable=train_model_logic
    )

    training_task