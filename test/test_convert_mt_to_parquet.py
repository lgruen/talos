"""
Test for ConvertMatrixTableToParquet.py
"""

import os
import tempfile
import polars as pl
import hail as hl

from talos.annotation_scripts.ConvertMatrixTableToParquet import flatten_mt, write_ht_to_parquet


def test_convert_mt_to_parquet():
    num_samples = 3
    num_variants = 10
    mt = hl.balding_nichols_model(n_populations=1, n_samples=num_samples, n_variants=num_variants)

    # Only keep the locus + alleles (row key, hence implicit) and GT entries for simplicity.
    mt = mt.select_rows().select_entries('GT')

    # Convert integer sample IDs to strings.
    mt = mt.annotate_cols(s=hl.str(mt.sample_idx))
    mt = mt.key_cols_by('s').drop('sample_idx')

    ht = flatten_mt(mt)
    assert ht.count() == num_variants

    with tempfile.TemporaryDirectory() as tmp_dir:
        write_ht_to_parquet(ht, os.path.join(tmp_dir, 'test'), compression='lz4')
        pq_df = pl.read_parquet(os.path.join(tmp_dir, 'test.parquet'))
        # columns: 2 * locus (contig, position), alleles, and 2 * GT (alleles, phased) for each sample
        assert pq_df.shape == (num_variants, 2 + 1 + 2 * num_samples)
