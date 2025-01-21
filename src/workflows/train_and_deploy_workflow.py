import mlrun
from kfp import dsl


@dsl.pipeline(name="GitOps Training Pipeline", description="Train a model")
def pipeline(
    source_url: str,
    label_column: str,
    allow_validation_failure: bool,
    test_size: float,
    model_name: str,
    sentiment_model: str,
    text_column: str,
    sentiment_column: str,
    ordinal_columns: list,
    drop_columns: list,
):
    # Get our project object
    project = mlrun.get_current_project()

    # Ingest data
    ingest = project.run_function(
        "data",
        handler="get_data",
        inputs={"data": source_url},
        outputs=["data"],
    )
    
    # Compute sentiment data
    sentiment_fn = project.get_function("data")
    sentiment_fn.with_requests(cpu=6)
    sentiment_fn.with_limits(cpu=6)
    sentiment = project.run_function(
        sentiment_fn,
        handler="sentiment_analysis",
        inputs={"data": ingest.outputs["data"]},
        params={
            "sentiment_model": sentiment_model,
            "text_column": text_column,
        },
        outputs=["data_w_sentiment"],
    )

    # Validate data integrity
    validate_data_integrity = project.run_function(
        "validate",
        handler="validate_data_integrity",
        inputs={"data": sentiment.outputs["data_w_sentiment"]},
        params={
            "label_column": label_column,
            "allow_validation_failure": allow_validation_failure,
        },
        outputs=["passed_suite"],
    )

    # Process data
    process = project.run_function(
        "data",
        handler="process_data",
        inputs={"data": sentiment.outputs["data_w_sentiment"]},
        params={
            "label_column": label_column,
            "test_size": test_size,
            "ordinal_columns": ordinal_columns,
            "drop_columns" : drop_columns,
            "sentiment_column" : sentiment_column
        },
        outputs=["train", "test"],
    ).after(validate_data_integrity)

    # Validate train test split
    validate_train_test_split = project.run_function(
        "validate",
        handler="validate_train_test_split",
        inputs={"train": process.outputs["train"], "test": process.outputs["test"]},
        params={
            "label_column": label_column,
            "allow_validation_failure": allow_validation_failure,
        },
        outputs=["passed_suite"],
    )

    train = project.run_function(
        "train",
        inputs={
            "train": process.outputs["train"],
            "test": process.outputs["test"],
        },
        params={
            "label_column": label_column,
            "model_name" : model_name
        },
        hyperparams={
            "bootstrap": [True, False],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        },
        selector="max.accuracy",
        hyper_param_options=mlrun.model.HyperParamOptions(
            strategy="random", max_iterations=5
        ),
        outputs=["model", "model_uri"],
    ).after(validate_train_test_split)

    validate_model = project.run_function(
        "validate",
        handler="validate_model",
        inputs={
            "train": process.outputs["train"],
            "test": process.outputs["test"],
        },
        params={
            "model_path": train.outputs["model_uri"],
            "label_column": label_column,
            "allow_validation_failure": allow_validation_failure,
        },
        outputs=["passed_suite"],
    )

    # Deploy model to endpoint
    serving_fn = project.get_function("serving")
    deploy = project.deploy_function(
        serving_fn,
        models=[
            {
                "key": model_name,
                "model_path": train.outputs["model_uri"],
                "class_name" : "ChurnModel"
            }
        ],
    ).after(validate_model)
