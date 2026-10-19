# File provenance

Every script in `src/` is a copy of the file that produced the published results in the authors' project.
[`FILE_PROVENANCE.csv`](FILE_PROVENANCE.csv) lists, for each published file:

- `original_project_file` — path in the authors' project;
- `sha256_original` / `sha256_published` — checksums before and after publication;
- `status` — *final (submitted lineage)*, *final (corrected revision lineage)*, or *submitted lineage; harmonised revision version pending*;
- `role` — what the script does in the paper.

The only edit made for publication is replacing absolute local path roots with placeholders
(`<DATA_ROOT>`, `<PROJECT_ROOT>`, `<DOWNLOADS>`, `<WORKSPACE>`, `<HOME>`). Where the two checksums are equal, the file
contained no such paths.

Earth Engine scripts in `gee/scripts/` are byte-identical to the scripts that were run (renamed for clarity; the original
names are kept in Git history).

Earlier exploratory and superseded scripts (thesis-era visualisations, abandoned forecasting variants such as GNN, CNN,
ARIMA/SARIMA and multivariate runs, the original single-site Tonzi Ranch analyses, intermediate audit tools, and old output
images) were removed from the current tree. They remain available in the Git history of this repository.
