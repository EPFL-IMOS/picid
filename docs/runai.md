# RunAI
We Provide Scripts to run the Project on a RunAI Cluster


```bash
source /workspace/setup/.venv/bin/activate
uv pip install -e ".[dev]"
python lmetk/run.py paths=runai debug=default experiment=railway_traction/autogluon_railway_fit_predict
```


# Several Debugging CMD Lines

```bash
python -c "from lmetk.data.datasources.base import phmd_loader; print(phmd_loader)"
python -c "import pkgutil, lmetk; print([name for _, name, _ in pkgutil.iter_modules(lmetk.__path__)])"
```