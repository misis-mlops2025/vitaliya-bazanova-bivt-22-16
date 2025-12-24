"""
Script for training models based on Hydra configuration.
"""
import logging
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
    """Factory function to initialize the model."""
    if cfg.model.name == "logistic_regression":
        return LogisticRegression(**cfg.model.params, random_state=cfg.random_state)
    if cfg.model.name == "random_forest":
        return RandomForestClassifier(**cfg.model.params, random_state=cfg.random_state)
    if cfg.model.name == "decision_tree":
        return DecisionTreeClassifier(**cfg.model.params, random_state=cfg.random_state)
    raise ValueError(f"Model {cfg.model.name} is not supported.")

def load_data(cfg: DictConfig):
    """Generates synthetic data."""
    X, y = make_classification(
        n_samples=cfg.dataset.n_samples,
        n_features=cfg.dataset.n_features,
        n_informative=cfg.dataset.n_informative,
        random_state=cfg.random_state
    )
    return train_test_split(
        X, y, test_size=cfg.dataset.test_size, random_state=cfg.random_state
    )

def run_training(cfg: DictConfig) -> float:
    """
    Main training logic detached from Hydra decorator for testing.
    """
    log.info("Starting training pipeline...")
    X_train, X_test, y_train, y_test = load_data(cfg)

    model = get_model(cfg)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    log.info(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def train(cfg: DictConfig):
    """Entry point for Hydra."""
    run_training(cfg)

if __name__ == "__main__":
    train() # pylint: disable=no-value-for-parameter