#!/usr/bin/env python3

"""
Creates a feature table from an annotated per-family table.

We extract additional features that a downstream ML model may need / that are more
suitable for e.g. random forest models.
"""

import argparse
import logging
import time
from contextlib import contextmanager
from pydantic import BaseModel, Field, ValidationError
import polars as pl

from talos.models import PhenotypeMatchedPanels, Pedigree
from talos.utils import TWO, make_flexible_pedigree, read_json_from_path


# Except if specifically mentioned, the features all relate to the proband. Potentially
# we could make this more explicit in the naming.
# TODO(leo): This needs to live in models.py or somewhere else where it can be shared
# between training and inference. We'll also want dedicated versioning for each schema
# (which models.py currently doesn't do).
class FeatureTableSchemaV1(BaseModel):
    """
    Schema for the feature table.
    """

    # The coordinates (except maybe for the chromosome) are not actually useful as
    # features, but are required as a unique row key to join any other tables (e.g.
    # labels for training).
    # TODO(leo): The `Coordinates` model implies that all our sites have been coverted
    # to be biallelic, i.e. there's only a single alternate allele. Instead of a nested
    # coordinates representation, we use flat column, which probably also compress better.
    chrom: str
    pos: int
    ref: str
    alt: str

    # Genotype related features. Note that hom-ref variants are never considered,
    # meaning that if `is_het` is False, the genotype is always hom-alt.
    is_het: bool = Field(..., description='Is the genotype heterozygous?')

    # Quality and depth metrics.
    genotype_quality: int = Field(..., description='Genotype quality (GQ)')
    read_depth: int = Field(..., description='Read depth (DP)')
    allele_balance: float = Field(..., description='Allele balance (AB)')

    # Family related features.
    sex: int = Field(..., description='Proband sex from pedigree')
    de_novo: bool = Field(..., description='Is the variant de novo?')

    # gnomAD
    gnomad_ac: int = Field(..., description='gnomAD allele count')
    gnomad_af: float = Field(..., description='gnomAD minor allele frequency')
    gnomad_ac_xy: int = Field(..., description='gnomAD alternate allele count for XY samples')


def validate_input(df: pl.DataFrame):
    # TODO(leo): We should validate the input format here, but we don't have a
    # dedicated schema yet

    # Let's at least verify that all sites are biallelic.
    non_biallelic_sites = df.filter(
        pl.col('alleles').list.len() != TWO,
    ).select(pl.col('locus.contig'), pl.col('locus.position'), pl.col('alleles'))

    if non_biallelic_sites.shape[0] > 0:
        raise ValueError(f'Input contains non-biallelic sites: {non_biallelic_sites}')


def validate_output(df: pl.DataFrame) -> None:
    """Validates output data against the schema.

    Raises a ValidationError if the data does not conform to the schema.
    """

    for record in df.to_dicts():
        try:
            # Try to create a FeatureTableSchemaV1 instance. If this fails, a
            # ValidationError will be raised.
            FeatureTableSchemaV1(**record)
        except ValidationError as e:
            raise ValidationError(f'Output validation failed for record {record}') from e


def extract_features(input_df: pl.DataFrame, pedigree: Pedigree) -> pl.DataFrame:
    """
    Extracts features from the input DataFrame.

    Args:
        input_df: Input DataFrame with annotated per-family table
        pedigree: Pedigree information

    Returns:
        A DataFrame with extracted features
    """

    # TODO(leo): need to pass this in, based on the per-family table
    proband_sample_id = 'proband'
    mother_sample_id = 'mother'
    father_sample_id = 'father'

    proband_sex = pedigree.by_id[proband_sample_id].sex

    # Aliases for simpler access.
    proband_alleles = pl.col(f'{proband_sample_id}.GT.alleles')
    proband_ad = pl.col(f'{proband_sample_id}.AD')
    mother_hom_ref = pl.col(f'{mother_sample_id}.GT.alleles').list.eval(pl.element() == 0).list.all()
    father_hom_ref = pl.col(f'{father_sample_id}.GT.alleles').list.eval(pl.element() == 0).list.all()

    # Filter out all sites where the proband is hom-ref.
    # Note that the alleles might be phased, so we need to check all GT entries.
    # TODO(leo): Once this has been moved upstream, we should move this to
    # the input validation checks.
    filtered_df = input_df.filter(proband_alleles.list.eval(pl.element() != 0).list.any())

    return filtered_df.select(
        # Genomic coordinates from locus and alleles
        pl.col('locus.contig').alias('chrom'),
        pl.col('locus.position').alias('pos'),
        pl.col('alleles').list.get(0).alias('ref'),
        pl.col('alleles').list.get(1).alias('alt'),
        # gnomAD related data
        pl.col('gnomad.gnomad_AC').alias('gnomad_ac'),
        pl.col('gnomad.gnomad_AF').alias('gnomad_af'),
        pl.col('gnomad.gnomad_AC_XY').alias('gnomad_ac_xy'),
        # Determine if heterozygous by checking if the allele list contains 0
        # (since hom-ref 0/0 is already filtered out)
        pl.col(f'{proband_sample_id}.GT.alleles').list.contains(0).alias('is_het'),
        # Quality and depth metrics.
        pl.col(f'{proband_sample_id}.GQ').alias('genotype_quality'),
        pl.col(f'{proband_sample_id}.DP').alias('read_depth'),
        (proband_ad.list.get(1) / proband_ad.list.sum()).alias('allele_balance'),
        # Family related features.
        pl.lit(proband_sex).alias('sex'),
        # TODO(leo): very simplified de-novo check: this should also take
        # quality etc. into account, but we could also leave that to the model.
        (mother_hom_ref & father_hom_ref).alias('de_novo'),
    )


# TODO(leo): could move to this to a separate utils file, as it's generally useful.
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


def main(
    input_path: str,
    pedigree_path: str,
    output_path: str,
    compression: str = 'zstd',
    participant_panels_path: str | None = None,
):
    """
    Reads an annotated per-family table, extracts features, and writes to a Parquet file.

    Args:
        input_path: Path to the input MatrixTable
        output_path: Path to write the output Parquet file
        compression: Compression codec to use (default: zstd)
    """

    with timed_section('Overall feature table creation'):
        with timed_section(f'Reading pedigree file from {pedigree_path} / {participant_panels_path}'):
            pheno_panels: PhenotypeMatchedPanels = read_json_from_path(
                participant_panels_path,
                return_model=PhenotypeMatchedPanels,
                default=PhenotypeMatchedPanels(),
            )
            pedigree = make_flexible_pedigree(pedigree_path, pheno_panels)

        with timed_section(f'Reading input Parquet file from {input_path}'):
            input_df = pl.read_parquet(input_path)

        logging.info(f'Input columns: {input_df.columns}')
        logging.info(f'Input shape: {input_df.shape}')

        with timed_section('Validating input against schema...'):
            validate_input(input_df)

        with timed_section('Extracting features'):
            features = extract_features(input_df, pedigree)

        logging.info(f'Extracted features shape: {features.shape}')
        logging.info(f'Extracted features columns: {features.columns}')

        with timed_section('Validating output against schema...'):
            validate_output(features)

        with timed_section(f'Writing output Parquet file to {output_path}.parquet with {compression} compression'):
            # polars doesn't expose the ParquetCompression type, so we can't do type checking here.
            features.write_parquet(f'{output_path}.parquet', compression=compression)  # type: ignore[arg-type]


def cli_main():
    """Parses arguments and runs the conversion."""
    parser = argparse.ArgumentParser(description='Create a per-family feature table')
    parser.add_argument('--input', required=True, help='Path to the input annotated per-family table')
    parser.add_argument('--pedigree', help='Path to PED file', required=True)
    parser.add_argument('--participant_panels', help='GeneratePanelData JSON', default=None)
    parser.add_argument(
        '--output',
        required=True,
        help='Path to write the output Parquet file (without .parquet extension)',
    )
    parser.add_argument('--compression', default='zstd', help='Compression codec to use')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    main(
        input_path=args.input,
        pedigree_path=args.pedigree,
        output_path=args.output,
        compression=args.compression,
        participant_panels_path=args.participant_panels,
    )


if __name__ == '__main__':
    cli_main()
