import pandas as pd
from datetime import datetime, timedelta
from feast import FeatureStore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_model():
    print("--- 1. Инициализация Feast ---")
    store = FeatureStore(repo_path=".")

    print("--- 2. Формирование Entity DataFrame (Запрос данных) ---")
    driver_ids = [1001, 1002, 1003, 1004, 1005]

    # Генерируем 10 дат (последние 10 дней)
    timestamps = [datetime.now() - timedelta(days=i) for i in range(10)]

    entity_rows = []
    for d_id in driver_ids:
        for ts in timestamps:
            entity_rows.append({
                "driver_id": d_id,
                "event_timestamp": ts
            })

    entity_df = pd.DataFrame.from_dict(entity_rows)

    print("--- 3. Получение исторических фичей из Feast ---")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate",
            "driver_hourly_stats:avg_daily_trips"
        ]
    ).to_df()

    training_df.dropna(inplace=True)

    print(f"Получено записей для обучения: {len(training_df)}")
    print(training_df.head())

    print("--- 4. Обучение модели ---")
    target = "avg_daily_trips"
    features = ["conv_rate", "acc_rate"]

    X = training_df[features]
    y = training_df[target]

    # Разбиваем и обучаем простую линейную регрессию
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Проверка
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Модель обучена! MSE: {mse}")

    return model


if __name__ == "__main__":
    train_model()