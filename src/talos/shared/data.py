"""Data loading and processing utilities shared between training and inference."""

import os
import logging
import time
from pathlib import Path
from contextlib import contextmanager

import polars as pl
from typing import Any

from talos.shared.validation import validate_dataframe, _compare_schemas
from talos.shared.schema import feature_schema, label_schema


@contextmanager
def timed_section(description: str):
    """
    Context manager for timing code sections and logging the result.

    Args:
        description: Description of the section being timed
    """
    logging.info(f'Starting: {description}')
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logging.info(f'Completed: {description} in {elapsed_time:.2f} seconds')


def safe_read_parquet(file_path: str, expected_schema: dict[str, pl.DataType], df_name: str) -> pl.DataFrame:
    """
    Reads a Parquet file and validates its schema against the expected schema.

    Args:
        file_path: Path to the Parquet file.
        expected_schema: The expected Polars dictionary schema.
        df_name: Name of the DataFrame type (e.g., "Feature", "Label") for error messages.

    Returns:
        DataFrame read from the Parquet file.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        ValueError: If the Parquet file schema does not match the expected schema or schema reading fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{df_name} Parquet file not found at: {file_path}')

    # We can't just use read_parquet_schema in advance, because that doesn't fill in the
    # enum categories, which we want to validate against.
    logging.debug(f'Reading {df_name} Parquet file: {file_path}')
    result = pl.read_parquet(file_path)
    schema = result.schema

    try:
        _compare_schemas(schema, expected_schema, f'{df_name} Parquet file schema ({file_path})')
    except ValueError as e:
        raise ValueError(f"Schema validation failed for {df_name} Parquet file '{file_path}': {e}") from e

    logging.debug(f'Schema validated for {df_name} file: {file_path}')
    return result


def safe_write_parquet(
    df: pl.DataFrame,
    file_path: str,
    expected_schema: dict[str, pl.DataType],
    df_name: str,
    **kwargs: Any,
) -> None:
    """
    Validates a DataFrame's schema against the expected schema using the shared validation
    logic before writing it to a Parquet file, passing extra arguments to write_parquet.

    Args:
        df: The DataFrame to write.
        file_path: The path to write the Parquet file to.
        expected_schema: The expected Polars dictionary schema.
        df_name: Name of the DataFrame type (e.g., "Feature", "Label") for error messages.

    Raises:
        ValueError: If the DataFrame schema does not match the expected schema.
    """
    try:
        validate_dataframe(df, expected_schema, f'{df_name} DataFrame')  # Use public function
        logging.debug(f'Schema validated for {df_name} DataFrame. Writing to: {file_path}')
        df.write_parquet(file_path, **kwargs)
    except ValueError as e:
        raise ValueError(f"Schema validation failed before writing {df_name} Parquet file to '{file_path}': {e}") from e


def load_feature_file(file_path: str) -> pl.DataFrame:
    """
    Safely load a feature file, validating its schema before reading.

    Args:
        file_path: Path to the feature Parquet file

    Returns:
        DataFrame containing features
    """
    return safe_read_parquet(file_path, feature_schema, 'feature')


def load_label_file(file_path: str) -> pl.DataFrame:
    """
    Safely load a label file, validating its schema before reading.

    Args:
        file_path: Path to the label Parquet file

    Returns:
        DataFrame containing labels
    """
    return safe_read_parquet(file_path, label_schema, 'Label')


def load_data_pair(feature_file: str, label_file: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load a pair of feature and label files.

    Args:
        feature_file: Path to the feature Parquet file
        label_file: Path to the label Parquet file

    Returns:
        Tuple of (features_df, labels_df)
    """
    with timed_section(f'Loading feature file: {feature_file}'):
        features_df = load_feature_file(feature_file)

    with timed_section(f'Loading label file: {label_file}'):
        labels_df = load_label_file(label_file)

    return features_df, labels_df


def match_feature_label_files(features_dir: str, labels_dir: str) -> list[tuple[str, str]]:
    """
    Match feature and label files by name pattern.

    Args:
        features_dir: Directory containing feature Parquet files
        labels_dir: Directory containing label Parquet files

    Returns:
        List of tuples of matched (feature_path, label_path)
    """
    feature_files = {
        Path(f).stem: os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.parquet')
    }

    label_files = {Path(f).stem: os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.parquet')}

    common_stems = set(feature_files.keys()) & set(label_files.keys())

    if not common_stems:
        raise ValueError(f'No matching feature/label files found between {features_dir} and {labels_dir}')

    # Sort the stems to ensure reproducible ordering
    return [(feature_files[stem], label_files[stem]) for stem in sorted(common_stems)]


def load_all_data(features_dir: str, labels_dir: str) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Load all matching feature and label files from the specified directories.

    Args:
        features_dir: Directory containing feature Parquet files
        labels_dir: Directory containing label Parquet files

    Returns:
        Dictionary mapping family ID to (features_df, labels_df) tuple
    """
    matched_files = match_feature_label_files(features_dir, labels_dir)

    result = {}
    for feature_path, label_path in matched_files:
        family_id = Path(feature_path).stem
        result[family_id] = load_data_pair(feature_path, label_path)

    return result


def join_features_with_labels(features: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """
    Join features with their corresponding labels.

    Args:
        features: DataFrame containing features
        labels: DataFrame containing labels

    Returns:
        DataFrame with features and labels joined
    """
    join_cols = ['chrom', 'pos', 'ref', 'alt']
    return features.join(
        labels,
        on=join_cols,
        how='inner',
    )
