#!/usr/bin/env python3

"""
Main training script for the Talos ML model.

This script loads feature and label data, prepares it for training,
trains a CatBoost model, and evaluates its performance.
"""

import argparse
import logging
import os
from typing import Any

import polars as pl
from catboost import CatBoostClassifier, Pool
import mlflow
from sklearn.model_selection import train_test_split
from pathlib import Path

from talos.shared.data import match_feature_label_files, load_data_pair, join_features_with_labels
from talos.shared.schema import feature_schema
from talos.training.preprocessing import identify_feature_types, downsample_negatives
from talos.training.evaluation import evaluate_model
from talos.training.mlflow_utils import (
    initialize_mlflow,
    log_model_params,
    log_model_metrics,
    log_model_artifact,
    log_dataset_info,
)


def train_catboost_model(
    train_data: pl.DataFrame,
    val_data: pl.DataFrame,
    feature_cols: list[str],
    categorical_features: list[str],
    target_col: str = 'is_causative',
    params: dict[str, Any] | None = None,
) -> CatBoostClassifier:
    """
    Train a CatBoost model.

    Args:
        train_data: Training data
        val_data: Validation data
        feature_cols: Feature column names
        categorical_features: Categorical feature names
        target_col: Target column name
        params: Model parameters

    Returns:
        Trained CatBoost model
    """
    # Set default parameters if not provided
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': 100,
            'early_stopping_rounds': 50,
        }

    # Extract feature matrices and target vectors
    x_train = train_data.select(feature_cols).to_numpy()
    y_train = train_data[target_col].to_numpy()

    x_val = val_data.select(feature_cols).to_numpy()
    y_val = val_data[target_col].to_numpy()

    # Get indices of categorical features
    categorical_indices = [feature_cols.index(feature) for feature in categorical_features]

    # Create CatBoost pools
    train_pool = Pool(
        data=x_train,
        label=y_train,
        cat_features=categorical_indices,
    )

    val_pool = Pool(
        data=x_val,
        label=y_val,
        cat_features=categorical_indices,
    )

    # Initialize and train the model
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        verbose=params.get('verbose', 100),
        early_stopping_rounds=params.get('early_stopping_rounds', 50),
    )

    logging.info(f'Model trained for {model.tree_count_} iterations')

    return model


# Helper function to load, join, and add family_id for a given family ID.
def _process_family(family_id: str, family_files: dict[str, tuple[str, str]]) -> pl.DataFrame | None:
    if family_id not in family_files:
        logging.warning(f'File paths not found for family {family_id}')
        return None
    feature_path, label_path = family_files[family_id]
    features_df, labels_df = load_data_pair(feature_path, label_path)
    if features_df is None or labels_df is None:
        logging.warning(f'Data loading failed for family {family_id} (paths: {feature_path}, {label_path})')
        return None
    # Join and add family ID
    joined_df = join_features_with_labels(features_df, labels_df)
    return joined_df.with_columns(pl.lit(family_id).alias('family_id'))


def _prepare_train_evaluate(
    features_dir: str,
    labels_dir: str,
    output_dir: str,
    test_size: float,
    val_size: float,
    negative_ratio: int,
    random_seed: int,
    catboost_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Loads data, prepares splits, trains model, and evaluates.

    Returns:
        Dictionary containing materialized results: train_data, val_data,
        test_data, train_families, model, evaluation_results, model_path,
        predictions_path, feature_cols, categorical_features.
    """

    # 1. Load and join data
    logging.info(f'Loading data from {features_dir} and {labels_dir}')
    matched_files = match_feature_label_files(features_dir, labels_dir)
    family_files = {Path(fp).stem: (fp, lp) for fp, lp in matched_files}
    family_ids = list(family_files.keys())

    # 2. Split families into train/val/test sets based on family IDs
    logging.info(f'Splitting {len(family_ids)} families into train/val/test sets')
    train_val_families, test_families = train_test_split(
        family_ids,
        test_size=test_size,
        random_state=random_seed,
    )
    # Adjust val_size calculation for splitting train_val_families
    adjusted_val_size = val_size / (1 - test_size) if (1 - test_size) > 0 else 0
    if adjusted_val_size >= 1.0 or adjusted_val_size <= 0:
        raise ValueError(
            f'Invalid adjusted validation size: {adjusted_val_size}. '
            f'Check test_size ({test_size}) and val_size ({val_size}).',
        )

    train_families, val_families = train_test_split(
        train_val_families,
        test_size=adjusted_val_size,
        random_state=random_seed,
    )

    logging.info(
        f'Split: {len(train_families)} train, {len(val_families)} validation, {len(test_families)} test families',
    )

    # 3. Process and concatenate data incrementally for each set
    logging.info('Processing and concatenating data...')
    feature_types = identify_feature_types(feature_schema)
    feature_cols = feature_types['categorical'] + feature_types['numerical']
    categorical_features = feature_types['categorical']

    # Process training data (with downsampling)
    train_data_list = []
    logging.info(f'Processing {len(train_families)} training families...')
    for _i, family_id in enumerate(train_families):
        family_df = _process_family(family_id, family_files)
        if family_df is not None:
            # Apply downsampling only to training data.
            downsampled_df = downsample_negatives(
                family_df,
                negative_to_positive_ratio=negative_ratio,
                random_seed=random_seed,
            )
            train_data_list.append(downsampled_df)
    train_data = pl.concat(train_data_list) if train_data_list else pl.DataFrame()
    logging.info(f'Finished processing training data. Final shape: {train_data.shape}')

    # Process Validation Data (no downsampling)
    val_data_list = []
    logging.info(f'Processing {len(val_families)} validation families...')
    for _i, family_id in enumerate(val_families):
        family_df = _process_family(family_id, family_files)
        if family_df is not None:
            val_data_list.append(family_df)
    val_data = pl.concat(val_data_list) if val_data_list else pl.DataFrame()
    logging.info(f'Finished processing validation data. Final shape: {val_data.shape}')

    # Process Test Data (no downsampling)
    test_data_list = []
    logging.info(f'Processing {len(test_families)} test families...')
    for _i, family_id in enumerate(test_families):
        family_df = _process_family(family_id, family_files)
        if family_df is not None:
            test_data_list.append(family_df)
    test_data = pl.concat(test_data_list) if test_data_list else pl.DataFrame()
    logging.info(f'Finished processing test data. Final shape: {test_data.shape}')

    # 4. Train the model
    logging.info('Training CatBoost model')
    model = train_catboost_model(
        train_data=train_data,
        val_data=val_data,
        feature_cols=feature_cols,
        categorical_features=categorical_features,
        target_col='is_causative',
        params=catboost_params,
    )
    model_path = os.path.join(output_dir, 'model.cbm')
    model.save_model(model_path)
    logging.info(f'Model saved to {model_path}')

    # 5. Evaluate the model
    logging.info('Evaluating model on test data')
    evaluation_results = evaluate_model(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        target_col='is_causative',
        family_col='family_id',
    )
    predictions_path = os.path.join(output_dir, 'test_predictions.parquet')
    evaluation_results['predictions'].write_parquet(predictions_path)
    logging.info(f'Test predictions saved to {predictions_path}')

    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_families': train_families,  # Needed for logging family count
        'model': model,
        'evaluation_results': evaluation_results,
        'model_path': model_path,
        'predictions_path': predictions_path,
        'feature_cols': feature_cols,
        'categorical_features': categorical_features,
    }


def run_training(
    features_dir: str,
    labels_dir: str,
    output_dir: str,
    mlflow_experiment_name: str,
    register_model: bool = False,
    model_name: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.15,
    negative_ratio: int = 100,
    random_seed: int = 42,
    catboost_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the complete training pipeline, logging results to MLflow.

    Args:
        features_dir: Directory containing feature Parquet files
        labels_dir: Directory containing label Parquet files
        output_dir: Directory to save model and artifacts
        mlflow_experiment_name: MLflow experiment name
        register_model: Whether to register the model in MLflow
        model_name: Name to use when registering the model
        test_size: Proportion of data to use for testing
        val_size: Proportion of remaining data to use for validation
        negative_ratio: Ratio of negative to positive examples for downsampling
        random_seed: Random seed
        catboost_params: Parameters for CatBoost model

    Returns:
        Dictionary containing paths to artifacts and evaluation metrics.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare data, train model, and evaluate (outside MLflow run)
    logging.info('Starting data preparation, model training, and evaluation...')
    prep_eval_results = _prepare_train_evaluate(
        features_dir=features_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        test_size=test_size,
        val_size=val_size,
        negative_ratio=negative_ratio,
        random_seed=random_seed,
        catboost_params=catboost_params,
    )
    logging.info('Data preparation, training, and evaluation completed.')

    # Extract results needed for logging and return value
    model = prep_eval_results['model']
    evaluation_results = prep_eval_results['evaluation_results']
    model_path = prep_eval_results['model_path']
    predictions_path = prep_eval_results['predictions_path']
    feature_cols = prep_eval_results['feature_cols']
    categorical_features = prep_eval_results['categorical_features']
    train_data = prep_eval_results['train_data']
    val_data = prep_eval_results['val_data']
    test_data = prep_eval_results['test_data']
    train_families = prep_eval_results['train_families']

    # 2. Initialize MLflow and log results
    initialize_mlflow(experiment_name=mlflow_experiment_name)
    logging.info(f'Starting MLflow run for experiment: {mlflow_experiment_name}')

    with mlflow.start_run():
        logging.info('Logging parameters, metrics, and artifacts to MLflow...')

        # Log data preparation parameters
        mlflow.log_param('data_split_test_size', test_size)
        mlflow.log_param('data_split_val_size', val_size)
        mlflow.log_param('data_split_random_seed', random_seed)
        mlflow.log_param('train_downsample_negative_ratio', negative_ratio)

        # Log dataset info
        train_pos_count = train_data.filter(pl.col('is_causative')).height if not train_data.is_empty() else 0
        train_neg_count = train_data.filter(~pl.col('is_causative')).height if not train_data.is_empty() else 0
        log_dataset_info(
            feature_cols=feature_cols,
            categorical_features=categorical_features,
            train_size=len(train_data),
            val_size=len(val_data),
            test_size=len(test_data),
            positive_count=train_pos_count,
            negative_count=train_neg_count,
            family_count=len(train_families),
        )

        # Log model parameters
        model_params = model.get_params()
        log_model_params(model_params)

        # Log evaluation metrics
        log_model_metrics(evaluation_results['metrics'])

        # Create an input example from a small subset of the validation data
        x_example = val_data.select(feature_cols).head(5).to_numpy() if not val_data.is_empty() else None

        # Log model artifact with input example
        log_model_artifact(
            model=model,  # Pass the model object as required
            artifact_path='model',
            registered_name=model_name if register_model else None,
            input_example=x_example,
        )

        # Log predictions artifact
        mlflow.log_artifact(predictions_path, artifact_path='predictions')

        logging.info('MLflow logging complete.')

    # Return paths and metrics, not large objects
    return {
        'metrics': evaluation_results['metrics'],
        'model_path': model_path,
        'predictions_path': predictions_path,
    }


def main():
    """Parse command line arguments and run training."""
    parser = argparse.ArgumentParser(description='Train Talos ML model')

    parser.add_argument(
        '--features-dir',
        required=True,
        help='Directory containing feature Parquet files',
    )
    parser.add_argument(
        '--labels-dir',
        required=True,
        help='Directory containing label Parquet files',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save model and artifacts',
    )
    parser.add_argument(
        '--mlflow-experiment-name',
        required=True,
        help='MLflow experiment name',
    )
    parser.add_argument(
        '--register-model',
        action='store_true',
        help='Register model in MLflow model registry',
    )
    parser.add_argument(
        '--model-name',
        default='talos-catboost',
        help='Name to use when registering the model',
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing',
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Proportion of remaining data to use for validation',
    )
    parser.add_argument(
        '--negative-ratio',
        type=int,
        default=100,
        help='Ratio of negative to positive examples for downsampling',
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Set up CatBoost parameters
    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': args.random_seed,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 100 if args.verbose else 20,
        'early_stopping_rounds': 50,
    }

    # Run training
    try:
        results = run_training(
            features_dir=args.features_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            mlflow_experiment_name=args.mlflow_experiment_name,
            register_model=args.register_model,
            model_name=args.model_name,
            test_size=args.test_size,
            val_size=args.val_size,
            negative_ratio=args.negative_ratio,
            random_seed=args.random_seed,
            catboost_params=catboost_params,
        )

        # Print summary metrics
        print('\nTraining completed successfully!')
        print('\nModel Metrics:')
        for metric, value in results['metrics'].items():
            if isinstance(value, int | float):
                print(f'  {metric}: {value:.4f}')

        print(f'\nModel saved to: {results["model_path"]}')

    except Exception as e:
        logging.exception(f'Training failed: {e}')
        raise


if __name__ == '__main__':
    main()
