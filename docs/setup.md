# How to install TabPHM

Please install `uv` first.

```
git submodule update --init --recursive
uv sync
uv pip install -e ".[dev]"
```

Now you can try to fit your first model:

either by using VSCode run config in `.vscode/launch.json`

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
