"""
Hail-specific utils -- split out to reduce import times for tools that don't need Hail.
"""

import hail as hl


def hail_table_from_tsv(tsv_file: str, new_ht: str, types: dict[str, hl.tstr] | None = None):
    """
    take a previously created TSV file and ingest it as a Hail Table
    requires an initiated Hail context

    Args:
        tsv_file ():
        new_ht ():
        types (dict[str, hl.tstr]): optional, a dictionary of column names and their types
    """

    if types is None:
        types = {}

    # import as a hail table, force=True as this isn't Block-Zipped so all read on one core
    # We also provide some data types for non-string columns
    ht = hl.import_table(tsv_file, types=types, force=True)

    # combine the two alleles into a single list
    ht = ht.transmute(locus=hl.locus(contig=ht.chrom, pos=ht.pos), alleles=[ht.ref, ht.alt])
    ht = ht.key_by('locus', 'alleles')
    ht.write(new_ht)
    ht.describe()
