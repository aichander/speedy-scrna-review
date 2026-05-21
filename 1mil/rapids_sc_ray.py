import ray

ray.init(num_gpus=4)

@ray.remote(num_gpus=1)
def process_scrna(data_pth: str, output_pth: str):
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    import rapids_singlecell as rsc
    import scanpy as sc
    import anndata as ad
    import zarr
    from packaging.version import parse as parse_version
    from pathlib import Path

    # RMM pool
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=10 * 1024**3,
        maximum_pool_size=14 * 1024**3,
        managed_memory=True,
    )
    cp.cuda.set_allocator(rmm_cupy_allocator)

    # Load based on file extension
    ext = Path(data_pth).suffix
    if ext == ".zarr":
        if parse_version(ad.__version__) < parse_version("0.12.0rc1"):
            from anndata.experimental import read_elem_as_dask as read_dask
        else:
            from anndata.experimental import read_elem_lazy as read_dask

        f = zarr.open(data_pth)
        X = f["X"]
        shape = X.attrs["shape"]
        adata = ad.AnnData(
            X=read_dask(X, (5_000, shape[1])),
            obs=ad.io.read_elem(f["obs"]),
            var=ad.io.read_elem(f["var"]),
        )
    elif ext in (".h5ad", ".h5"):
        adata = sc.read_h5ad(data_pth)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    # Process
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.calculate_qc_metrics(adata)
    rsc.pp.normalize_total(adata)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.25)
    rsc.pp.scale(adata, max_value=10, zero_center=False)
    rsc.pp.pca(adata, svd_solver="arpack")
    adata.obsm["X_pca"] = adata.obsm["X_pca"].compute().get()
    rsc.pp.neighbors(adata, n_neighbors=50, use_rep="X_pca", n_pcs=20)
    rsc.tl.umap(adata, min_dist=0.45, init_pos="spectral", random_state=0, n_components=2)
    rsc.tl.leiden(adata, resolution=1, n_iterations=2)

    # Save based on output extension
    out_ext = Path(output_pth).suffix
    if out_ext == ".zarr":
        adata.write_zarr(output_pth)
    elif out_ext in (".h5ad", ".h5"):
        adata.write(output_pth)
    else:
        raise ValueError(f"Unsupported output format: {out_ext}")

    return output_pth


# # Zarr in, h5ad out
# ray.get(process_scrna.remote("adata.zarr", "result.h5ad"))

# # h5ad in, zarr out
# ray.get(process_scrna.remote("adata.h5ad", "result.zarr"))

# results = ray.get(futures)

ray.shutdown()