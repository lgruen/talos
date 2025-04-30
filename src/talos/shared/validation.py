"""Schema validation functions for feature and label data."""

import polars as pl
from talos.shared.schema import feature_schema, label_schema


def _compare_schemas(
    actual_schema: dict[str, pl.DataType],
    expected_schema: dict[str, pl.DataType],
    source_description: str,
) -> None:
    """
    Compares an actual schema dictionary against an expected Polars dictionary schema
    with strict type checking.

    Args:
        actual_schema: The schema dictionary ({col_name: PolarsType}) to validate.
        expected_schema: The expected Polars dictionary schema to validate against.
        source_description: Description of the source for error messages
        (e.g., "Feature DataFrame", "Parquet file schema").

    Raises:
        ValueError: If columns are missing, extra, or have incorrect data types.
    """
    expected_columns = set(expected_schema.keys())
    actual_columns = set(actual_schema.keys())

    # Check for missing columns
    missing_columns = expected_columns - actual_columns
    if missing_columns:
        raise ValueError(f'{source_description} is missing required columns: {missing_columns}')

    # Check for extra columns
    extra_columns = actual_columns - expected_columns
    if extra_columns:
        raise ValueError(f'{source_description} has unexpected extra columns: {extra_columns}')

    # Check column types strictly
    type_mismatches = []
    for col_name, expected_type in expected_schema.items():
        actual_type = actual_schema[col_name]
        if actual_type != expected_type:
            type_mismatches.append(f"Column '{col_name}': Expected type {expected_type}, found {actual_type}")

    if type_mismatches:
        raise ValueError(f'{source_description} has type mismatches:\n' + '\n'.join(type_mismatches))


def validate_dataframe(df: pl.DataFrame, expected_schema: dict[str, pl.DataType], df_name: str) -> None:
    """
    Validates a DataFrame's schema against a given Polars dictionary schema.

    Args:
        df: DataFrame containing the data to validate.
        expected_schema: The Polars dictionary schema to validate against.
        df_name: Name of the DataFrame type (e.g., "Feature", "Label") for error messages.

    Raises:
        ValueError: If columns are missing, extra, or have incorrect data types.
    """
    _compare_schemas(df.schema, expected_schema, f'{df_name} DataFrame')


def validate_features(df: pl.DataFrame) -> None:
    """
    Validates feature data against the feature_schema schema.

    Args:
        df: DataFrame containing feature data.

    Raises:
        ValueError: If validation fails.
    """
    validate_dataframe(df, feature_schema, 'Feature')


def validate_labels(df: pl.DataFrame) -> None:
    """
    Validates label data against the label_schema schema.

    Args:
        df: DataFrame containing label data.

    Raises:
        ValueError: If validation fails.
    """
    validate_dataframe(df, label_schema, 'Label')
