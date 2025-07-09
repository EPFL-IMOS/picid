import itertools
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
import wandb


def tag_filter(tags, filter_out_tags):
    for t in tags:
        if t in filter_out_tags:
            return True
    return False


def process_run_wide(run, cfg, experiment_aliases, errors, categories):
    """
    Creates a dataframe in wide format with columns:
    - exp_name: experiment alias
    - seed: seed of the experiment
    - errors as columns
    """

    print(f"Processing run: exp_name {cfg.get('exp_name', 'unknown')}")

    keys = {}
    for e, c in itertools.product(errors, categories):
        key = f"{c}/{e}"
        if key not in run.summary.keys():
            print(f"Error: {c}/{e} not found in run.summary {run}")
        else:
            keys[key] = run.summary[key] if e != "mape" else run.summary[key] * 100

    df = pd.DataFrame(keys, index=[0])
    df["seed"] = cfg.seed
    df["exp_name"] = experiment_aliases[cfg.exp_name]
    df.set_index(["exp_name", "seed"], inplace=True)
    df.set_index(["exp_name"], inplace=True)
    return df

def process_run_long(run, cfg, experiment_aliases, errors, categories):
    """
    Creates a dataframe in long format with columns:
    - name: experiment alias
    - seed: seed of the experiment
    - category: category of the error (e.g., "P", "Q", "U", "I")
    - error: type of error (e.g., "nmae", "nrmse", "mape")
    - value: value of the error metric
    """
    
    df = []
    for category, error in itertools.product(categories, errors):
        q_string = f"{category}/{error}"
        if q_string not in run.summary.keys():
            print(f"Error: {q_string} not found in run.summary {run}")
        else:
            value = run.summary[q_string] if error != "mape" else run.summary[q_string] * 100
            df.append({
                "name": experiment_aliases[cfg.exp_name], 
                "seed": cfg.seed, 
                "category": category, 
                "error": error, 
                "value": value
            })
    
    return pd.DataFrame(df)

class WandBEvaluation:


    def __init__(self, 
                 entity: str, 
                 project: str,
                 filter_out_tags: list = None,
                 experiment_aliases: dict = None,
                 debug: bool = False,
                 tags: list = None,
                 errors: list = [],
                 categories: list = []):
        
        self.entity = entity
        self.project = project
        self.tags = tags if tags is not None else []
        self.filter_out_tags = filter_out_tags if filter_out_tags is not None else []
        self.experiment_aliases = experiment_aliases if experiment_aliases is not None else {}
        self.is_initialized = False
        self.debug = debug

        self.erros = errors
        self.categories = categories

    def initialize(self):
        self.api = wandb.Api()
        self.is_initialized = True
        return self


    def load_runs(self, dataframe_format="long"):

        assert dataframe_format in ["wide", "long"], "dataframe_format must be either 'wide' or 'long'"

        if dataframe_format == "wide":
            process_run = process_run_wide
        else:
            process_run = process_run_long

        if not self.is_initialized:
            raise RuntimeError("WandBEvaluation is not initialized. Call initialize() first.")

        experiments = list(self.experiment_aliases.keys())

        wandb_runs = self.api.runs(f"{self.entity}/{self.project}")
        runs = []
        for run in tqdm(wandb_runs):
            cfg = OmegaConf.create(run.config)
            # print(f"Checking run: exp_name {cfg.get('exp_name', 'unknown')}")

            if (
                run.state == "finished"
                and ("exp_name" in cfg)
                and cfg.exp_name in experiments
                and not tag_filter(run.tags, self.filter_out_tags)
                and tag_filter(run.tags, self.tags)
            ):
                runs.append(process_run(run, cfg, self.experiment_aliases, self.erros, self.categories))

        rdf = pd.concat(runs).sort_index().copy()
        self.rdf = rdf
        return self
              
    def to_latex(self):


        # header = r"""
        # & \multicolumn{2}{l}{P} & \multicolumn{2}{l}{Q}  & \multicolumn{2}{l}{U}  & \multicolumn{2}{l}{I} \\
        # \cmidrule(l){2-3} \cmidrule(l){4-5} \cmidrule(l){6-7} \cmidrule(l){8-9} 
        # Ablation & NMAE & NRMSE & NMAE & NRMSE  & NMAE & NRMSE  & NMAE & NRMSE     \\
        # """

        if not hasattr(self, 'rdf'):
            raise RuntimeError("No runs loaded. Call load_runs() first.")
        
        # Convert the DataFrame to LaTeX format
        latex_code = self.rdf.to_latex(escape=False, index=True)
        # lines = latex_code.splitlines()

        # TODO: Modify the header to include the new header later
        # toprule_index = lines.index("\\toprule")
        # midrule_index = lines.index("\\midrule")
        # modified_latex_code = "\n".join(lines[:toprule_index+1] + header.splitlines()[:] + lines[midrule_index:])
        return latex_code