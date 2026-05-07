# RunAI

To run on RunAI (or a general Kubernetes cluster), we provide a `Dockerfile` and setup scripts under `container/`.

Steps:
- Customize the scripts with your setup / credentials.
- `build_container.sh` - create the image
- `push_container.sh` - push to registry
- `create_container_runai.sh` - run the container on the cluster



> **Warning**
> We assume that you have access to persistent storage mounted to the docker container. An example:
> ```bash
> -v ~/datasets:/mnt/storage/${LDAP_USERNAME}/datasets \
> -v ~/Experiments:/mnt/storage/${LDAP_USERNAME}/experiments \
> -v ~/Workspaces/work/PICID:/mnt/storage/${LDAP_USERNAME}/projects
> ```
> These folders are symlinked to /workspace/datasets, ... in the container due to restrictions in how many PVCs can be mounted on the cluster.

Once the container is started, the following steps are necessary to run:

```bash
tmux
mkdir -p $HOME/lib
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 $HOME/lib/libcuda.so
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
source /workspace/setup/.venv/bin/activate
cd /workspace/projects/PICID
uv sync --group dev
# uv pip install --upgrade netCDF4 h5py xarray nvitop
python picid/run.py paths=runai experiment=railway_traction/forecasting/tabpfn_fit_predict
python picid/run.py paths=runai experiment=concepts_n_cmapss_ds02/prognostics/tabdpt_fit_predict

```

Alternatively, we provide a runscript `runai.sh`.

```bash
chmod +x runai.sh
./runai.sh debug=default experiment=railway_traction/forecasting/tabpfn_fit_predict
```

For local installation and running experiments without RunAI, see [Setup](../getting-started/setup.md).

[← Guides index](index.md) | [Back to documentation index](../index.md)
