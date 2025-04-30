process CreateFeatureTable {
    container params.container
    publishDir params.cohort_output_dir, mode: 'copy'

    input:
    path family_parquet_table

    output:
    path "${params.cohort}_features.parquet"

    script:
    """
        set -ex
        
        CreateFeatureTable \
            --input \${params.family_parquet_table} \
            --output ${params.cohort}_features.parquet
        """
}
