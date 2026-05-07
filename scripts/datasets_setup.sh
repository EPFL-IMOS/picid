# mkdir -p datasets/

# # railway
# mkdir -p railway/
# gdown --id 1F8-fDdazvoAvS5gl9bNZrTZHzJn6E7pW -O datasets/railway/railway.csv

# # railway_agg
# gdown --id 1RNG0moIpNp0LCXCtn-BS_NkGgJqM5CCI -O datasets/railway_agg.csv


# ncmapps_ds02
#gdown --id 1qcsUl84Vxl5sXBeRoLDGnf9SEl72BFBe -O datasets/ncmapps_ds02.csv

# # building
# gdown --id 1rSyzyJWkjcnBjegac34WVPlFVg6UkqBZ -O datasets/alpha_building_years_02_03_occupancy_levelwise.csv


# # Download datasets
# gdown https://drive.google.com/uc?id=1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot
# unzip iTransformer_datasets.zip -d datasets/
# mv datasets/iTransformer_datasets/* datasets/
# rm -r datasets/iTransformer_datasets
# rm iTransformer_datasets.zip


# mkdir -p datasets/N-CMAPSS
#gdown --folder https://drive.google.com/drive/folders/1iWx09Pam7aCra5-C4Qba15kgUFqtwBGk -O datasets/N-CMAPSS

# IMPORTANT: PHME20 is required for phme20 pipeline tests, snapshot generation, and experiments.
# The phmd library can auto-download on first load, but that fails in CI and is slow for new setups.
# Pre-downloading here ensures datasets/phmd_cache/datasets/ is populated before any loader runs.
#uv run python -c "from phmd.download import download; download('PHME20', cache_dir='datasets/phmd_cache', unzip=True)"
