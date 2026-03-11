"""
Tests for picid_report.config.

Covers PipelineConfig and from_default().
"""

from picid_report import config
from picid_report.config import PipelineConfig


class TestPipelineConfig:
    """PipelineConfig: bundled config for pipeline overrides."""

    def test_from_default_returns_instance(self):
        pc = PipelineConfig.from_default()
        assert isinstance(pc, PipelineConfig)
        assert pc.column_config is not None
        assert isinstance(pc.column_config, dict)
        assert "model_target" in pc.column_config
        assert pc.expected_search_space is not None
        assert isinstance(pc.expected_search_space, dict)
        assert isinstance(pc.column_filters_to_ignore_for_hp_search, list)
        assert isinstance(pc.special_columns, list)
        assert isinstance(pc.columns_to_normalize, list)
        assert isinstance(pc.column_filters_to_drop, list)

    def test_from_default_search_space_structure(self):
        pc = PipelineConfig.from_default()
        # May be empty or model -> {hp: [values]}
        for k, v in pc.expected_search_space.items():
            assert isinstance(v, dict)
            for kk, vv in v.items():
                assert isinstance(vv, list), f"expected list of values for {k}/{kk}"

    def test_from_default_sort_metric_resolver_optional(self):
        pc = PipelineConfig.from_default()
        # Resolver may be set if configs available, or None
        assert pc.sort_metric_resolver is None or callable(pc.sort_metric_resolver)

    def test_manual_construction(self):
        pc = PipelineConfig(
            column_config=dict(config.COLUMN_CONFIG),
            expected_search_space={"M1": {"lr": [0.01, 0.001]}},
            column_filters_to_ignore_for_hp_search=list(
                config.COLUMN_FILTERS_TO_IGNORE_FOR_HP_SEARCH
            ),
            special_columns=list(config.SPECIAL_COLUMNS),
            columns_to_normalize=list(config.COLUMNS_TO_NORMALIZE),
            column_filters_to_drop=list(config.COLUMN_FILTERS_TO_DROP),
            sort_metric_resolver=None,
        )
        assert pc.expected_search_space["M1"]["lr"] == [0.01, 0.001]
