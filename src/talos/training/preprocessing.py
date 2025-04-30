"""Data preprocessing pipeline for model training."""

import logging
import polars as pl


def identify_feature_types(schema: pl.Schema) -> dict[str, list[str]]:
    """
    Returns a dictionary with keys 'categorical' and 'numerical', mapping to lists of
    column names defined in the current feature schema.
    """
    categorical_features = []
    numerical_features = []

    # Iterate through the fields defined in the schema dictionary
    for field_name, field_type in schema.items():
        # Skip coordinate fields as they are not features for the model
        if field_name in {'chrom', 'pos', 'ref', 'alt'}:
            continue

        # Determine feature type based on Polars DataType
        if isinstance(field_type, pl.Categorical | pl.Enum | pl.Boolean):
            categorical_features.append(field_name)
        elif isinstance(
            field_type,
            pl.Int32 | pl.Float32 | pl.Int64 | pl.Float64,
        ):  # Add other potential numerical types if needed
            numerical_features.append(field_name)
        else:
            raise TypeError(f"Unexpected feature type '{field_type}' for field '{field_name}'. ")

    return {
        'categorical': categorical_features,
        'numerical': numerical_features,
    }


def downsample_negatives(
    df: pl.DataFrame,
    negative_to_positive_ratio: int,
    random_seed: int,
    target_col: str = 'is_causative',
) -> pl.DataFrame:
    """
    Downsample negative examples in a DataFrame to a target ratio relative to positives.

    Args:
        df: Input DataFrame.
        negative_to_positive_ratio: Target ratio of negative to positive examples.
        random_seed: Random seed for reproducibility.
        target_col: Name of the boolean target column indicating positive examples.

    Returns:
        DataFrame with negative examples downsampled.
    """
    positives = df.filter(pl.col(target_col))
    negatives = df.filter(~pl.col(target_col))

    n_positives = positives.height
    n_negatives = negatives.height

    if n_positives == 0:
        raise ValueError('No positive examples found in the input DataFrame. Cannot downsample negatives.')

    if n_negatives == 0:
        # No negatives to downsample
        return df

    # Calculate target number of negatives
    target_n_negatives = min(n_negatives, n_positives * negative_to_positive_ratio)

    if target_n_negatives >= n_negatives:
        # No downsampling needed as current negative count is already at or below the target
        return df

    logging.debug(f'Downsampling negatives from {n_negatives} to {target_n_negatives}')
    negatives_downsampled = negatives.sample(
        n=target_n_negatives,
        seed=random_seed,
        shuffle=True,
    )
    return pl.concat([positives, negatives_downsampled])
