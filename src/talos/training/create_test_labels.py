#!/usr/bin/env python3
"""
Script to create a labels parquet file from a features parquet file.
Sets is_causative=True for variants matching the provided chrom-pos-ref-alt patterns.
"""

import argparse
import sys
from pathlib import Path

import polars as pl


# Constants
EXPECTED_PARTS = 4


def parse_variant_string(variant_str):
    """Parse a variant string in the format chrom-pos-ref-alt."""
    try:
        parts = variant_str.split('-')
        if len(parts) != EXPECTED_PARTS:
            raise ValueError(f"Invalid variant format: {variant_str}. Expected format: chrom-pos-ref-alt")

        chrom, pos, ref, alt = parts
        # Convert position to integer
        pos = int(pos)
        return chrom, pos, ref, alt
    except ValueError as e:
        print(f"Error parsing variant {variant_str}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Create labels file with is_causative=True for matching variants')
    parser.add_argument('variants', nargs='+', help='Variants in format chrom-pos-ref-alt')
    parser.add_argument('--input', required=True, help='Input features parquet file')
    parser.add_argument('--output', required=True, help='Output labels parquet file')
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse variant strings
    variants_to_match = [parse_variant_string(v) for v in args.variants]

    # Read the features file
    features_df = pl.scan_parquet(args.input)

    # Create a new column 'is_causative' that's True if the row matches any of the variants
    expr = pl.lit(False)
    for chrom, pos, ref, alt in variants_to_match:
        # Build an expression that checks if this row matches the current variant
        match_expr = (
            (pl.col("chrom") == chrom) &
            (pl.col("pos") == pos) &
            (pl.col("ref") == ref) &
            (pl.col("alt") == alt)
        )
        # Combine with OR to match any of the variants
        expr = expr | match_expr

    # Create the output dataframe with is_causative and the key columns needed for joining
    result_df = features_df.select(
        "chrom",
        "pos",
        "ref",
        "alt",
        expr.alias("is_causative"),
    ).collect()

    # Write the output
    result_df.write_parquet(args.output)

    # Print summary
    match_count = result_df.filter(pl.col("is_causative")).height
    print(f"Created labels file with {match_count} of {result_df.height} variants marked as causative")


if __name__ == "__main__":
    main()
