from dataclasses import dataclass

@dataclass(frozen=True)
class TransformationArtifact:
    preprocessor_path: str
    train_npz_path: str
    test_npz_path: str
    feature_names_path: str
