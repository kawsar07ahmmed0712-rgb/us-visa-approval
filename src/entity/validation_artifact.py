from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationArtifact:
    report_path: str
    missing_report_path: str
    schema_path: str
    status: bool
