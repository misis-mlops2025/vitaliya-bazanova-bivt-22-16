import logging
import sys
import os
import pickle
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Настраиваем простой логгер
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def predict(config_path="conf/config.yaml", input_file=None):
    """
    Loads model and makes predictions on a specific file.
    If input_file is provided, it overrides the config path.
    """
    # 1. Загружаем конфиг, чтобы знать, где лежит модель
    # Мы используем путь внутри Docker, так как код будет запускаться там
    cfg = OmegaConf.load(config_path)
    model_path = cfg.dataset.output_model_path  # Это /opt/airflow/models/model.pkl

    # 2. Читаем данные
    try:
        df = pd.read_csv(input_file)
        if 'target' in df.columns:
            X = df.drop(columns=['target'])
        else:
            X = df
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return

    # 3. Загружаем модель
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # 4. Предсказываем
    predictions = model.predict(X)

    # 5. Сохраняем результат
    output_file = input_file.replace(".csv", "_predictions.csv")
    pd.DataFrame(predictions, columns=['prediction']).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")



if __name__ == "__main__":
    # Позволяем передавать путь к файлу через аргументы командной строки
    # python src/models/predict_model.py /opt/data/new_data.csv
    input_csv = sys.argv[1] if len(sys.argv) > 1 else None
    predict(input_file=input_csv)