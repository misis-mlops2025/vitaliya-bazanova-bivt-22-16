import logging
import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.datasets import make_classification

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def make_dataset(cfg: DictConfig):
    log.info("Generating dataset...")
    X, y = make_classification(
        n_samples=cfg.dataset.n_samples,
        n_features=cfg.dataset.n_features,
        n_informative=cfg.dataset.n_informative,
        random_state=cfg.random_state
    )

    # Собираем в DataFrame для удобства сохранения
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    df['target'] = y

    output_path = cfg.dataset.raw_data_path
    df.to_csv(output_path, index=False)
    log.info(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    make_dataset()