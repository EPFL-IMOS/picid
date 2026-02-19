# How to install TabPHM

Please install `uv` first.
On Windows, you need to have [build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed (dependency of numpoly).

Furthermore make sure that your git key is setup on github as this
is required to clone submodules in the first step.

```
git submodule update --init --recursive
uv sync
uv pip install -e ".[dev]"
```

Now you can try to fit your first model:

either by using VSCode run config in `.vscode/launch.json`

# Setup Local Paths
To run the project, you have to adjust your system paths.
For this step you can create a new file in `configs/paths`.
You can copy the [default](../../configs/paths/default.yaml).

## How to Run an Experiment

```json
{
    "name": "Railway Traction: Fit-Predict Autogluon",
    "type": "debugpy",
    "request": "launch",
    "program": "lmetk/run.py",
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
python lmetk/run.py paths=rt_local experiment=railway_traction/autogluon_railway_fit_predict
```

If you want to make the run faster, use debug mode to develop:
```bash
source .venv/bin/activate
python lmetk/run.py paths=rt_local debug=default experiment=railway_traction/autogluon_railway_fit_predict
```

You can also customize your paths locally using a config file in `configs/paths`

```bash
source .venv/bin/activate
python lmetk/run.py paths=rt_local debug=default experiment=railway_traction/autogluon_railway_fit_predict
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
