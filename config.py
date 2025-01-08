from pydantic import BaseModel


class MainConfig(BaseModel):
    source_url: str = "store://datasets/mlrun-quick-start/heart#0:latest"
    label_column: str = "target"
    allow_validation_failure: bool = True
    ohe_columns: list = ["sex", "cp", "slope", "thal", "restecg"]
    test_size: float = 0.2

workflow_configs = {"main": MainConfig()}
