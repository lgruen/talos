"""Evaluation metrics for model training and validation."""

from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)


def calculate_standard_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, float]:
    """
    Calculate standard classification metrics.

    Args:
        y_true: Array of true labels (0/1 or False/True)
        y_pred_proba: Array of predicted probabilities

    Returns:
        Dictionary of metric names to values
    """
    # Area under PR curve
    ap = average_precision_score(y_true, y_pred_proba)

    metrics = {
        'average_precision': ap,
    }

    # Calculate metrics at different thresholds
    for threshold in [0.1, 0.5, 0.9]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='binary',
            zero_division=0,
        )
        metrics.update(
            {
                f'precision_at_{threshold}': precision,
                f'recall_at_{threshold}': recall,
                f'f1_at_{threshold}': f1,
            },
        )

    return metrics


def calculate_top_k_metrics(
    df: pl.DataFrame,
    prob_col: str = 'prediction',
    target_col: str = 'is_causative',
    family_col: str = 'family_id',
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Calculate top-k accuracy metrics across families.

    Args:
        df: DataFrame with predictions and target
        prob_col: Column name for predicted probabilities
        target_col: Column name for target variable
        family_col: Column name for family grouping
        k_values: List of k values to calculate metrics for

    Returns:
        Dictionary of metric names to values
    """
    if k_values is None:
        k_values = [1, 5, 10]
    metrics = {}

    # Group by family and calculate top-k accuracy
    total_families = df[family_col].n_unique()
    families_with_positives = 0
    families_with_true_positive = dict.fromkeys(k_values, 0)

    # Process each family
    for family_id in df[family_col].unique():
        # Get family data and sort by prediction score
        family_df = df.filter(pl.col(family_col) == family_id).sort(prob_col, descending=True)

        # Check if there are any positives in this family
        has_positives = family_df[target_col].sum() > 0
        if has_positives:
            families_with_positives += 1

            # Check if a positive appears in the top-k
            for k in k_values:
                top_k = family_df.head(k)
                if top_k[target_col].sum() > 0:
                    families_with_true_positive[k] += 1

    # Calculate metrics
    for k in k_values:
        accuracy = families_with_true_positive[k] / families_with_positives if families_with_positives > 0 else 0.0
        metrics[f'top_{k}_accuracy'] = accuracy

    metrics['families_with_positives'] = families_with_positives
    metrics['total_families'] = total_families

    return metrics


def evaluate_model(
    model: Any,
    test_data: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = 'is_causative',
    family_col: str = 'family_id',
) -> dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model with predict_proba method
        test_data: Test data as Polars DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        family_col: Family ID column name

    Returns:
        Dictionary of metrics and predictions
    """
    # Extract features and target from polars DataFrame
    x_test = test_data.select(feature_cols).to_numpy()
    y_test = test_data[target_col].to_numpy()

    # Get predictions
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    # Add predictions to DataFrame
    test_data_with_preds = test_data.with_columns(pl.lit(y_pred_proba).alias('prediction'))

    # Calculate standard metrics
    standard_metrics = calculate_standard_metrics(y_test, y_pred_proba)

    # Calculate top-k metrics
    topk_metrics = calculate_top_k_metrics(
        test_data_with_preds,
        prob_col='prediction',
        target_col=target_col,
        family_col=family_col,
    )

    # Combine metrics
    metrics = {**standard_metrics, **topk_metrics}

    return {
        'metrics': metrics,
        'predictions': test_data_with_preds,
    }
