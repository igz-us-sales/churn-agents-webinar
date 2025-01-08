import mlrun
import pandas as pd
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn import ensemble


@mlrun.handler(outputs=["model_uri"])
def train_model(
    context: mlrun.MLClientCtx,
    train: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    bootstrap: bool,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    n_estimators: int,
    model_name: str,
):
    # X, y split
    X_train = train.drop(label_column, axis=1)
    y_train = train[label_column]
    X_test = test.drop(label_column, axis=1)
    y_test = test[label_column]

    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier(
        bootstrap=bootstrap,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
    )

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model, model_name=model_name, x_test=X_test, y_test=y_test)

    # Train our model
    model.fit(X_train, y_train)

    # Log model artifact URI for serving
    project = context.get_project_object()
    model_artifact = project.list_artifacts(
        name=model_name,
        iter=1,
        tag="latest"
    ).to_objects()[0]
    return model_artifact.uri