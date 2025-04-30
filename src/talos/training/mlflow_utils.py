"""MLflow integration utilities for model training and tracking."""

import logging
from typing import Any

import mlflow


def initialize_mlflow(experiment_name: str) -> str:
    """
    Initialize MLflow tracking.

    Args:
        experiment_name: Name for the experiment
        model_type: Type of model being trained

    Returns:
        The experiment name
    """
    # Get or create the experiment
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    logging.info(f'MLflow initialized with experiment: {experiment_name}')
    return experiment_name


def log_model_params(params: dict[str, Any]) -> None:
    """
    Log model parameters to MLflow.

    Args:
        params: Dictionary of parameter names and values
    """
    for name, value in params.items():
        mlflow.log_param(name, value)


def log_model_metrics(metrics: dict[str, float]) -> None:
    """
    Log model metrics to MLflow.

    Args:
        metrics: Dictionary of metric names and values
    """
    for name, value in metrics.items():
        if isinstance(value, int | float):
            mlflow.log_metric(name, value)
        else:
            logging.warning(f'Skipping non-numeric metric: {name}={value}')


def log_model_artifact(
    model: Any,
    artifact_path: str = 'model',
    registered_name: str | None = None,
    input_example: Any = None,
) -> None:
    """
    Log model artifact to MLflow and optionally register in model registry.

    Args:
        model: Trained model object
        artifact_path: Path to save the model artifact
        registered_name: If provided, register model with this name
        input_example: Sample input data for the model, used to infer schema
    """
    # Log model artifact
    mlflow.sklearn.log_model(
        model,
        artifact_path,
        input_example=input_example,
    )

    # Register model if name provided
    if registered_name:
        # Get run ID
        run_id = mlflow.active_run().info.run_id

        # Register model in MLflow model registry
        model_uri = f'runs:/{run_id}/{artifact_path}'
        result = mlflow.register_model(model_uri, registered_name)

        logging.info(f'Model registered as: {registered_name} (version {result.version})')


def log_dataset_info(
    feature_cols: list[str],
    categorical_features: list[str],
    train_size: int,
    val_size: int,
    test_size: int,
    positive_count: int,
    negative_count: int,
    family_count: int,
) -> None:
    """
    Log dataset information to MLflow.

    Args:
        feature_cols: List of all feature column names
        categorical_features: List of categorical feature names
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        positive_count: Number of positive samples in training set
        negative_count: Number of negative samples in training set
        family_count: Number of unique families
    """
    # Log dataset metrics
    mlflow.log_param('feature_count', len(feature_cols))
    mlflow.log_param('categorical_feature_count', len(categorical_features))
    mlflow.log_param('numerical_feature_count', len(feature_cols) - len(categorical_features))

    mlflow.log_param('train_samples', train_size)
    mlflow.log_param('val_samples', val_size)
    mlflow.log_param('test_samples', test_size)

    mlflow.log_param('positive_samples', positive_count)
    mlflow.log_param('negative_samples', negative_count)
    mlflow.log_param('pos_to_neg_ratio', positive_count / max(negative_count, 1))

    mlflow.log_param('family_count', family_count)

    # Create and log feature lists as artifacts
    with open('feature_list.txt', 'w') as f:
        f.write('\n'.join(feature_cols))

    with open('categorical_features.txt', 'w') as f:
        f.write('\n'.join(categorical_features))

    mlflow.log_artifact('feature_list.txt')
    mlflow.log_artifact('categorical_features.txt')
