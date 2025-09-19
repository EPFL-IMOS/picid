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
You can copy the [default](configs/paths/default.yaml).

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
