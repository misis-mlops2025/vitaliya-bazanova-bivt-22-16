"""
Script for training models based on Hydra configuration.
"""
import logging
import sys
import hydra
from omegaconf import DictConfig
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Настраиваем логгер
log = logging.getLogger(__name__)


def get_model(cfg: DictConfig):
    """
    Factory function to initialize the model based on config name.
    """
    if cfg.model.name == "logistic_regression":
        return LogisticRegression(**cfg.model.params, random_state=cfg.random_state)
    if cfg.model.name == "random_forest":
        return RandomForestClassifier(**cfg.model.params, random_state=cfg.random_state)
    if cfg.model.name == "decision_tree":
        return DecisionTreeClassifier(**cfg.model.params, random_state=cfg.random_state)
    raise ValueError(f"Model {cfg.model.name} is not supported.")


def load_data(cfg: DictConfig):
    """
    Generates synthetic data using sklearn make_classification.
    """
    X, y = make_classification(
        n_samples=cfg.dataset.n_samples,
        n_features=cfg.dataset.n_features,
        n_informative=cfg.dataset.n_informative,
        random_state=cfg.random_state
    )
    return train_test_split(
        X, y, test_size=cfg.dataset.test_size, random_state=cfg.random_state
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig) -> float:
    """
    Main training pipeline.
    """
    log.info("Starting training pipeline...")

    # 1. Генерация данных
    X_train, X_test, y_train, y_test = load_data(cfg)
    log.info(f"Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 2. Инициализация модели
    model = get_model(cfg)
    log.info(f"Initialized model: {cfg.model.name}")

    # 3. Обучение
    model.fit(X_train, y_train)
    log.info("Model trained.")

    # 4. Валидация
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    log.info(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter