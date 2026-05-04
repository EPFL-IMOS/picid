"""Guard against running tabfm models when the tabfm dependency group is absent."""

from __future__ import annotations

import importlib.util

from omegaconf import DictConfig

TABFM_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "picid.model.estimators.tabpfn": ("tabpfn", "TabPFN"),
    "picid.model.estimators.tabdpt": ("tabdpt", "TabDPT"),
    "picid.model.estimators.carte": ("carte_ai", "carte-ai"),
}


def check_tabfm_available(cfg: DictConfig) -> None:
    target: str = cfg.get("model", {}).get("_target_", "") or ""
    for module_prefix, (import_name, display_name) in TABFM_REQUIREMENTS.items():
        if module_prefix not in target:
            continue
        if importlib.util.find_spec(import_name) is None:
            raise RuntimeError(
                f"\n\n"
                f"  Model requires '{display_name}', which is not installed.\n"
                f"  This package belongs to the 'tabfm' dependency group.\n\n"
                f"  Install it with:\n"
                f"    uv sync --group tabfm\n\n"
                f"  Or exclude tabfm models by not using targets under:\n"
                f"    {module_prefix}\n"
            )
