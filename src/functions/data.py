import mlrun
import pandas
import pandas as pd
import pandera as pa
from datasets import Dataset, load_dataset
from pandera.typing import DataFrame
from sklearn import set_config
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoTokenizer, RobertaForSequenceClassification, pipeline

set_config(transform_output="pandas")


@mlrun.handler(outputs=["data"])
def get_data(data: pd.DataFrame):
    return data.copy()


@mlrun.handler(outputs=["data_w_sentiment"])
def sentiment_analysis(
    data: pd.DataFrame,
    sentiment_model: str,
    text_column: str
):
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
    model = RobertaForSequenceClassification.from_pretrained(sentiment_model)
    sentiment_classifier = pipeline(
        task="sentiment-analysis",
        tokenizer=tokenizer,
        model=model,
        top_k=1,
    )

    def sentiment(rows):
        resp = sentiment_classifier(rows[text_column])
        return {
            "sentiment_label": [i[0]["label"] for i in resp],
            "sentiment_score": [i[0]["score"] for i in resp],
        }

    data_w_sentiment = Dataset.from_pandas(data).map(
        sentiment, batched=True, batch_size=50
    )

    return data_w_sentiment.to_pandas()


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
            (
                OrdinalEncoder(categories=[["negative", "neutral", "positive"]]),
                [sentiment_column],
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
