"""Doc guards for the canonical model layout pages."""

from __future__ import annotations

from pathlib import Path


def test_model_capabilities_page_mentions_canonical_namespaces():
    text = Path("docs/modules/modeling/model-capabilities.md").read_text(
        encoding="utf-8"
    )
    assert "picid.model.estimators" in text
    assert "picid.model.forecasters" in text
    assert "picid.model.adapters.base" in text
