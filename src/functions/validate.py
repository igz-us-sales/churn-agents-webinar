import cloudpickle
import mlrun
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import (data_integrity, model_evaluation,
                                       train_test_validation)


def create_deepchecks_dataset(df: pd.DataFrame, label_column: str) -> Dataset:
    # Get numerical and categorical columns based on dataframe types
    numerical_columns = list(
        set(df.select_dtypes("number").columns) - set([label_column])
    )
    categorical_columns = list(
        set(df.select_dtypes("object").columns) - set([label_column])
    )
    return Dataset(df=df, label=label_column, cat_features=categorical_columns)


@mlrun.handler()
def validate_data_integrity(
    context: mlrun.MLClientCtx,
    data: pd.DataFrame,
    label_column: str,
    allow_validation_failure: bool = False,
):
    # Create deepchecks dataset with column metadata
    dataset = create_deepchecks_dataset(df=data, label_column=label_column)

    # Run suite
    data_integrity_suite = data_integrity()
    suite_result = data_integrity_suite.run(dataset)

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()

    context.log_result("passed_suite", passed_suite)
    context.log_artifact("suite_report", local_path=suite_report)

    assert (
        allow_validation_failure or passed_suite == True
    ), "Data integrity validation failed"


@mlrun.handler()
def validate_train_test_split(
    context: mlrun.MLClientCtx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    allow_validation_failure: bool = False,
):
    # Create deepchecks dataset with column metadata
    train_dataset = create_deepchecks_dataset(df=train, label_column=label_column)
    test_dataset = create_deepchecks_dataset(df=test, label_column=label_column)

    # Run suite
    train_test_validation_suite = train_test_validation()
    suite_result = train_test_validation_suite.run(
        train_dataset=train_dataset, test_dataset=test_dataset
    )

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()

    context.log_result("passed_suite", passed_suite)
    context.log_artifact("suite_report", local_path=suite_report)

    assert (
        allow_validation_failure or passed_suite == True
    ), "Train test split validation failed"


@mlrun.handler()
def validate_model(
    context: mlrun.MLClientCtx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_path: str,
    label_column: str,
    allow_validation_failure: bool = False,
):
    # Load model
    model_file, *_ = mlrun.artifacts.get_model(model_path)
    with open(model_file, "rb") as f:
        model = cloudpickle.load(f)

    # Create deepchecks dataset with column metadata
    train_dataset = create_deepchecks_dataset(df=train, label_column=label_column)
    test_dataset = create_deepchecks_dataset(df=test, label_column=label_column)

    # Run suite
    model_evaluation_suite = model_evaluation()
    suite_result = model_evaluation_suite.run(
        train_dataset=train_dataset, test_dataset=test_dataset, model=model
    )

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()

    context.log_result("passed_suite", passed_suite)
    context.log_artifact("suite_report", local_path=suite_report)

    assert allow_validation_failure or passed_suite == True, "Model validation failed"
