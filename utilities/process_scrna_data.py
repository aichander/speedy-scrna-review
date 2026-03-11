# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (ndmm-scrna)
#     language: python
#     name: ndmm-scrna
# ---

# %%
def process_adata(adata,
                  subset_hvgs=False,
                  run_harmony=False,
                  harmony_key=["sample.sampleKitGuid"],
                  resolution=1.0,
                  run_rank_genes=False):

    import scanpy as sc
    import scanpy.external as sce

    # Number of CPUs
    sc.settings.n_jobs = 30

    # This function will only run if the data is raw. If not, recover the raw data
    if adata.raw is None:
        already_normalized = 'log1p' in adata.uns
        if not already_normalized:
            import scipy.sparse as sps
            sample = adata.X[:100, :100]
            if sps.issparse(sample):
                sample = sample.toarray()
            already_normalized = sample.max() < 50

        if already_normalized:
            print(
                "No adata.raw found and X appears already normalized. Cannot recover raw counts.")
            return None
        else:
            print(
                "No adata.raw found. X appears to be raw counts — assigning to adata.raw.")
            adata.raw = adata

    # raw re-assign
    adata = adata.raw.to_adata()
    adata.raw = adata

    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )

    # normalize and log1p
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # HVG selection
    # Seurat defaults: filters by expression mean (0.0125–3) and dispersion (>0.25)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.25)
    # Optional: Subset to HVGs
    if subset_hvgs:
        adata = adata[:, adata.var['highly_variable']].copy()

    # Scale and PCA
    sc.pp.scale(adata, max_value=10, zero_center=False)
    sc.tl.pca(adata, n_comps=20, svd_solver="arpack")
    adata.obsm["X_pca_temp"] = adata.obsm["X_pca"]

    # Optional: Harmonize on a metadata parameter
    if run_harmony:
        sce.pp.harmony_integrate(adata, key=harmony_key)
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]

    # Neighbors and dimensionality reduction
    # n_neighbors=50: standard for immune cell atlases; reduce for finer populations
    sc.pp.neighbors(adata, n_neighbors=50, use_rep="X_pca", n_pcs=20)
    sc.tl.tsne(adata, n_pcs=20)
    
    # UMAP min_dist=0.45 preserves local structure; random_state=0 for reproducibility
    sc.tl.umap(adata, min_dist=0.45, random_state=0, n_components=2)
    
    # Leiden  resolution=1.0 default; n_iterations=2 for speed vs full convergence
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)

    # Optional: Rank genes (leiden)
    if run_rank_genes:
        sc.tl.rank_genes_groups(
            adata, groupby="leiden", method="t-test", corr_method="benjamini-hochberg"
        )

    return adata
