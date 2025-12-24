"""
Tests for the training pipeline.
"""
from omegaconf import OmegaConf
from src.models.train_model import get_model, load_data
from sklearn.linear_model import LogisticRegression
from src.models.train_model import run_training

def test_load_data():
    """Test data generation shapes."""
    cfg = OmegaConf.create({
        "dataset": {
            "n_samples": 100,
            "n_features": 10,
            "n_informative": 5,
            "test_size": 0.2
        },
        "random_state": 42
    })
    X_train, X_test, y_train, y_test = load_data(cfg)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert X_train.shape[1] == 10

def test_get_model_logistic():
    """Test model initialization."""
    cfg = OmegaConf.create({
        "model": {
            "name": "logistic_regression",
            "params": {"C": 0.5}
        },
        "random_state": 42
    })
    model = get_model(cfg)
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.5



def test_run_training():
    """Test the full training pipeline execution."""
    cfg = OmegaConf.create({
        "dataset": {
            "n_samples": 50,
            "n_features": 5,
            "n_informative": 3,
            "test_size": 0.2
        },
        "model": {
            "name": "logistic_regression",
            "params": {"C": 1.0}
        },
        "random_state": 42
    })
    accuracy = run_training(cfg)
    assert 0 <= accuracy <= 1.0