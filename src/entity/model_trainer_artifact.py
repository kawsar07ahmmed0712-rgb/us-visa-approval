from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTrainerArtifact:
    model_path: str
    metrics_path: str
    best_model_name: str
