@dataclass(frozen=True)
class Info:
    ds: str
    subexp: str
    type: str

DATASETS_PROGNOSTICS = [
    Info(ds="nb14", subexp="raw", type="prognostics"),
    Info(ds="nb14", subexp="combined", type="prognostics"),
    Info(ds="unibo", subexp="combined", type="prognostics"),
    Info(ds="phme20", subexp="raw", type="prognostics"),
    Info(ds="concepts_n_cmapss", subexp="", type="prognostics"),
    Info(ds="concepts_n_cmapss_ds02", subexp="", type="prognostics"),
    Info(ds="pronostia", subexp="raw", type="prognostics"),
    Info(ds="pronostia", subexp="combined", type="prognostics"),
    Info(ds="xjtu_sy", subexp="combined", type="prognostics"),
]

DATASETS_DIAGNOSTICS = [
    Info(ds="mzvav", subexp="", type="diagnostics"),
    Info(ds="hsf15/accumulator", subexp="", type="diagnostics"),  # hardest task
    Info(ds="hsf15/cooler", subexp="", type="diagnostics"),
    Info(ds="hsf15/pump", subexp="", type="diagnostics"),
    Info(ds="hsf15/valve", subexp="", type="diagnostics"),
    Info(ds="concepts_n_cmapss_multi", subexp="", type="diagnostics"),
]


def get_default_transforms(dataset, task, model):
    if 'hsf15' in dataset:
        dataset = '/'.join(dataset.split('_'))

    if task == 'prognostics':
        to_search = DATASETS_PROGNOSTICS
    elif task == 'diagnostics':
        to_search = DATASETS_DIAGNOSTICS
    else:
        assert False, (f"Task {task} not recognized (only prognostics and "
                       f"diagnostics have default transforms).")

