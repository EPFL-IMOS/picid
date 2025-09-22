# RunAI

To run on RunAI (or a general Kubernetes cluster), we provide a [Dockerfile](Dockerfile) and
scripts to setup in [container](../container/).

Steps:
- Customize the scripts with your setup / credentials.
- `build_container.sh` - create the image
- `push_container.sh` - push to registry
- `create_container_runai.sh` - run the container on the cluster



> **Warning**  
> We assume that you have access to persistent storage mounted to the docker container. An example:
> ```bash
> -v ~/datasets:/mnt/imos-students/${LDAP_USERNAME}/datasets \
> -v ~/Experiments:/mnt/imos-students/${LDAP_USERNAME}/experiments \
> -v ~/Workspaces/work/TabPHM:/mnt/imos-students/${LDAP_USERNAME}/projects
> ```
> These folders are symlinked to /workspace/datasets, ... in the container due to restrictions in how many pvc can be mounted on the EPFL RunAI cluster.

Once the container is started, the following steps are necessary to run:

```bash
source /workspace/setup/.venv/bin/activate
cd /workspace/projects/TabPHM
uv pip install -e ".[dev]"
python lmetk/run.py paths=runai debug=default experiment=railway_traction/autogluon_railway_fit_predict
```

Alternatively, we provide a runscript `runai.sh`.

```bash
chmod +x runai.sh
./runai.sh debug=default experiment=railway_traction/autogluon_railway_fit_predict
```

<!-- 
# Several Debugging CMD Lines

```bash
python -c "from lmetk.data.datasources.base import phmd_loader; print(phmd_loader)"
python -c "import pkgutil, lmetk; print([name for _, name, _ in pkgutil.iter_modules(lmetk.__path__)])"
``` -->