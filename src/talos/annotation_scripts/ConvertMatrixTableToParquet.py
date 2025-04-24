#!/usr/bin/env python3

"""
Converts a Hail MatrixTable to a Parquet file.

This script takes a Hail MatrixTable and converts it to a Parquet file,
preserving the row fields and flattening entry fields into columns per sample.
"""

import argparse
import logging

import hail as hl


def flatten_mt(mt: hl.MatrixTable) -> hl.Table:
    """
    Flattens a MatrixTable into a Table by flattening entry fields into columns per sample.

    This used to be handled by `make_table`, but that is now deprecated.

    Each column is named with the format `sample_id.field_name`, where `sample_id` is the
    sample ID and `field_name` is the name of the entry field.

    Args:
        mt: The input MatrixTable

    Returns:
        A Table with flattened entry fields per sample
    """
    if not (len(mt.col_key) == 1 and mt.col_key[0].dtype == hl.tstr):
        raise ValueError('column key must be a single field of type str')

    samples = mt.col_key[0].collect()
    localized = mt.localize_entries(entries_array_field_name='entries')
    return localized.annotate(
        **{
            f'{sample}.{field_name}': localized.entries[i][j]
            for i, sample in enumerate(samples)
            for j, field_name in enumerate(mt.entry)
        },
    ).drop('entries')


def write_ht_to_parquet(ht: hl.Table, output_path: str, compression: str = 'zstd'):
    """
    Writes a Hail Table to a Parquet file with specified compression.

    Args:
        ht: The input Table
        output_path: Path to write the output Parquet file
        compression: Compression codec to use (default: zstd)
    """
    # Write to Parquet with specified compression
    logging.info(
        f'Writing Parquet file to {output_path}.parquet with {compression} compression')
    ht.to_spark().write.option('compression', compression).parquet(
        f'{output_path}.parquet')
    logging.info('Conversion complete')


def main(
    input_path: str,
    output_path: str,
    compression: str = 'zstd',
):
    """
    Reads a Hail MatrixTable from disk and write it to Parquet with specified compression.

    Args:
        input_path: Path to the input MatrixTable
        output_path: Path to write the output Parquet file
        compression: Compression codec to use (default: zstd)
    """
    logging.info(f'Reading MatrixTable from {input_path}')
    mt = hl.read_matrix_table(input_path)
    n_samples = mt.count_cols()
    logging.info(f'MatrixTable contains {n_samples} samples')

    # Convert to table (this flattens the MatrixTable, creating columns for each sample's entry fields)
    logging.info(
        'Converting MatrixTable to table (flattening entries to columns)')
    ht = flatten_mt(mt)

    write_ht_to_parquet(ht, output_path, compression)


def cli_main():
    """Parses arguments and runs the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert a Hail MatrixTable to a Parquet file')
    parser.add_argument('--input', required=True,
                        help='Path to the input MatrixTable')
    parser.add_argument(
        '--output',
        required=True,
        help='Path to write the output Parquet file (without .parquet extension)',
    )
    parser.add_argument('--compression', default='zstd',
                        help='Compression codec to use')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Run the conversion
    main(
        input_path=args.input,
        output_path=args.output,
        compression=args.compression,
    )


if __name__ == '__main__':
    cli_main()
