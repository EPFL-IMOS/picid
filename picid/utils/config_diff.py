"""Compare two resolved config YAMLs and report key-level differences."""

from __future__ import annotations


def _flatten(cfg: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def diff_configs(path_a: str, path_b: str) -> dict:
    """
    Diff two YAML config files and return a structured report.

    Parameters
    ----------
    path_a : str
        Path to the first config file.
    path_b : str
        Path to the second config file.

    Returns
    -------
    dict with keys:
        ``only_in_first`` – list of flat keys present only in config A.
        ``only_in_second`` – list of flat keys present only in config B.
        ``different_values`` – list of (key, value_a, value_b) tuples.
    """
    from omegaconf import OmegaConf

    a = OmegaConf.load(path_a)
    b = OmegaConf.load(path_b)
    fa = _flatten(OmegaConf.to_container(a, resolve=True))  # type: ignore[arg-type]
    fb = _flatten(OmegaConf.to_container(b, resolve=True))  # type: ignore[arg-type]

    only_first = [k for k in fa if k not in fb]
    only_second = [k for k in fb if k not in fa]
    different = [(k, fa[k], fb[k]) for k in fa if k in fb and fa[k] != fb[k]]

    return {
        "only_in_first": only_first,
        "only_in_second": only_second,
        "different_values": different,
    }
