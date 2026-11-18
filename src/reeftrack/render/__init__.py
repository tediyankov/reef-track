"""Novel-view rendering of a run's reef (3D Gaussian Splatting / NeRF), trained off-site on a GPU.

* `colmap_export`  run -> a self-contained COLMAP-format scene folder (images + cameras + points)
                   that WaterSplatting (nerfstudio) and I2-NeRF (ZipNeRF) both read unchanged.
* `compare`        held-out ground-truth frames next to each method's render of the same camera.

See docs/rendering_jasmin.md for the full workflow on JASMIN.
"""
