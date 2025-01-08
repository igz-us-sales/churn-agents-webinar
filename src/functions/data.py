import mlrun
import pandas as pd
import pandas
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
# from src.functions.models import HeartSchema
import pandera as pa
from pandera.typing import DataFrame

from sklearn import set_config
set_config(transform_output = "pandas")

# @mlrun.handler(outputs=["data"])
# @pa.check_types(lazy=True)
# def get_data(data: pd.DataFrame) -> DataFrame[HeartSchema]:
#     # Alternatively
#     # HeartSchema.validate(data, lazy=True)
#     return data.copy()

@mlrun.handler(outputs=["data"])
def get_data(data: pd.DataFrame):
    return data.copy()


@mlrun.handler(outputs=["train", "test", "preprocessor:object"])
def process_data(
    data: pd.DataFrame,
    label_column: str,
    test_size: float,
    sentiment_column: str,
    ordinal_columns: list = None,
    drop_columns: list = None,
    random_state: int = 42,
):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)

    # Remove label before transforming
    y_train = train.pop(label_column)
    y_test = test.pop(label_column)
    
    ordinal_columns = ordinal_columns or []
    preprocessor = make_pipeline(
        make_column_transformer(
            # (OneHotEncoder(sparse_output=False), ["state"]),
            (OrdinalEncoder(), ordinal_columns),
            (OrdinalEncoder(
                categories=[["negative", "neutral", "positive"]]), [sentiment_column]
            ),
            ("drop", drop_columns),
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
    )
    
    preprocessor.fit(train)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)
    
    # Re-add label after transforming
    train[label_column] = y_train
    test[label_column] = y_test

    return train, test, preprocessor
