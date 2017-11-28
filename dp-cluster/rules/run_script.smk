# Snakemake rules for running scripts.
rule alg2_numpy:
    input:
        test_data('data/1d-cluster-{numCluster}.tsv')
    output:
        'figures/alg2_numpy/1d-cluster-{numCluster}.png'
    params:
        num_iteration = 50,
        cluster_variance = lambda wildcards: 1 / int(wildcards.numCluster) ** 2
    shell:
        'python scripts/alg2_numpy.py '
        '-i {input} '
        '-n {params.num_iteration} '
        '-c {params.cluster_variance} '
        '-v'