from dataclasses import dataclass

@dataclass(frozen=True)
class ModelEvaluationArtifact:
    evaluation_report_path: str
