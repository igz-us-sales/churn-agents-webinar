import pandera as pa
from pandera.typing import DataFrame, Series

class HeartSchema(pa.DataFrameModel):
    age: Series[int]
    sex: Series[str]
    cp: Series[str]
    exang: Series[int]
    fbs: Series[int]
    slope: Series[str]
    thal: Series[str]
    trestbps: Series[int]
    chol: Series[int]
    restecg: Series[str]
    thalach: Series[int]
    oldpeak: Series[float]
    ca: Series[float]
    target: Series[int] = pa.Field(in_range={"min_value": 0, "max_value": 1})