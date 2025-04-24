process ConvertMatrixTableToParquet {
    container params.container

    input:
        path(matrix_table_tar)

    publishDir params.cohort_output_dir, mode: 'copy'

    output:
        path("${params.cohort}.parquet")

    script:
        """
        set -ex
        tar --no-same-owner -xf ${matrix_table_tar}
        
        # Extract the MatrixTable directory name from the tar file
        MT_DIR=\$(basename ${matrix_table_tar} .tar)
        
        # Convert MatrixTable to Parquet
        ConvertMatrixTableToParquet \
            --input \${MT_DIR} \
            --output ${params.cohort}
        """
}
