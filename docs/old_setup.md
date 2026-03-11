# Legacy setup (RunAI / Guild)

> **Note:** This page describes an older workflow using Guild and rclone for RunAI. For the current RunAI workflow (Docker, container scripts, `runai.sh`), see [RunAI](guides/runai.md).

---

# How to RunAI
1. Start pod
2. kubectl port-forward test-torch2-v1 2022:2022 8080:8080
3. Access shell:  runai exec -it test-torch2-v1 bash
4. copy source code:  rclone sync . runai:/projects/load_management_evaluation  --exclude ".git/"  --verbose
    WARNING, be careful with rclone to not override local changes
5. wandb login <key>
6. pip install tabpfn lightning
7. Setup output directory structure:
    root
        guild_home (e.g. guild)
        exp_output (e.g. runs)


# How to run on RunAI


Configure Guild
 
```
wandb login <key>
export GUILD_HOME=/workspace/experiments/...
export MAX_DEFAULT_SOURCE_CODE_COUNT=999
```

Check source code or run an experiment
```
guild run experiment_runner:train exp_name=etth1 --test-sourcecode
guild run experiment_runner:train exp_name=etth1

```

```
on server 
guild run experiment_runner_server:train exp_name=etth1
```



Use Queue 

```
guild run --background --yes queue gpus=0
guild run --force-flags --stage-trials experiment_runner:sweep exp_name=etth1
```






## Usage 

1. Install Pytorch and necessary dependencies.

```
conda create -n cetst_env
conda activate cetst_env
conda install python=3.10 
conda install pytorch==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

