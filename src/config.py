from dataclasses import dataclass

@dataclass(frozen=True)
class DatasetConfig:
    csv_path: str

@dataclass(frozen=True)
class TrainingConfig:
    target_column: str
    positive_class: str
    negative_class: str
    test_size: float
    random_state: int

@dataclass(frozen=True)
class ArtifactsConfig:
    root_dir: str

@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig

def build_config(cfg: dict) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(**cfg["dataset"]),
        training=TrainingConfig(**cfg["training"]),
        artifacts=ArtifactsConfig(**cfg["artifacts"]),
    )
