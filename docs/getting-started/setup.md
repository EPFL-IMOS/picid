# How to install TabPHM

Please install `uv` first.
On Windows, you need to have [build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed (dependency of numpoly).

## Submodules

TabPHM uses git submodules in `local_packages/`. All are required for full functionality:

| Submodule | URL | Protocol |
|-----------|-----|----------|
| TabPFN | `git@github.com:EPFL-IMOS/TabPFN-IMOS.git` | SSH |
| phmd | `https://github.com/EPFL-IMOS/imos_phmd.git` | HTTPS |
| TabDPT | `https://github.com/layer6ai-labs/TabDPT.git` | HTTPS |
| carte-ai | `https://github.com/soda-inria/carte.git` | HTTPS |

**TabPFN uses SSH** (`git@...`)—ensure your [GitHub SSH key is configured](https://docs.github.com/en/authentication/connecting-to-github-with-ssh). The others use HTTPS. After cloning the repo, run:

```
git submodule update --init --recursive
uv sync
uv pip install -e ".[dev]"
```

Now you can try to fit your first model:

either by using VSCode run config in `.vscode/launch.json`

# Setup Local Paths

Configs resolve paths using `PROJECT_ROOT` (or the current working directory when run from the project root). The `picid` package sets `PROJECT_ROOT` automatically when imported.

To run the project, you must create a path config in `configs/paths/`. Copy [default.yaml](../../configs/paths/default.yaml) and set `data_dir`, `root_dir`, etc. for your system. Path configs define where data and artifacts live.

## Dataset Access

Datasets are stored under the path specified in your path config (typically `${paths.root_dir}/datasets`). **Datasets are not bundled**—users need to obtain them separately. Access and download links depend on the dataset:

- **phmd** and others have their own sources and may require registration or specific access.
- See individual datasource documentation or dataset provider websites for download and access instructions.

## How to Run an Experiment

```json
{
    "name": "Railway Traction: Fit-Predict Autogluon",
    "type": "debugpy",
    "request": "launch",
    "program": "picid/run.py",
    "console": "integratedTerminal",
    "args": [
        "paths=rt_local",
        "experiment=railway_traction/autogluon_railway_fit_predict",
        "debug=default"
    ],
    "env": {
        "HYDRA_FULL_ERROR": "1",
        "PYTHONFAULTHANDLER": "1"
    }
}
```

or by using the terminal

```bash
source .venv/bin/activate
python picid/run.py paths=rt_local experiment=railway_traction/autogluon_railway_fit_predict
```

If you want to make the run faster, use debug mode to develop:
```bash
source .venv/bin/activate
python picid/run.py paths=rt_local debug=default experiment=railway_traction/autogluon_railway_fit_predict
```

You can also customize your paths locally using a config file in `configs/paths`

```bash
source .venv/bin/activate
python picid/run.py paths=rt_local debug=default experiment=railway_traction/autogluon_railway_fit_predict
```


UPDATE: what has worked for L. on 17/10/2025

```bash
# Step 1: Clone all submodules defined in .gitmodules
git submodule update --init --recursive

# Step 2: Create the virtual environment
uv venv

# Step 3: Activate the new environment
source .venv/bin/activate

# Step 4: Install all dependencies from the lock file
uv sync

# This installs your project, dev dependencies, and all local submodules
uv pip install -e ".[dev]"
```

# Remove env and all dependencies 

```bash
# Deactivate if you're in the .venv
deactivate

# Remove the virtual environment folder
rm -rf .venv

# Forcefully unregister all submodules and remove their directories
git submodule deinit --all -f
git clean -fdx local_packages
```
