# Phased Refactoring Plan

**✅ REFACTORING COMPLETE** - All phases successfully completed on 2026-02-06

## Overview

**✅ ALL PHASES COMPLETE** - This refactoring has been successfully completed.

**Completion Summary**:
- ✅ All 5 phases implemented and tested
- ✅ 194 tests passing (Phase 0 + test_preprocess, test_utils, test_run, test_config added; see REFACTORING_PLAN_V2.md for current status)
- ✅ Backward compatibility maintained throughout
- ✅ Comprehensive documentation added
- ✅ Migration guide provided for users

This plan outlines the refactoring that was completed:
1. ✅ **Phase 1 COMPLETE**: Reorganize configs into `configs/` folder with new structure
2. ✅ **Phase 2 COMPLETE**: Add sort metric functionality to reporting (sort/rank by different metric)
3. ✅ **Phase 3 COMPLETE**: Integrate sort metric resolution into analysis (automatic resolution from configs)
4. ✅ **Phase 4 COMPLETE**: Migrate `EXPECTED_SEARCH_SPACE` to dataset/model structure
5. ✅ **Phase 5 COMPLETE**: Cleanup and documentation

**Final Status**: 194 tests passing ✅ | All functionality implemented ✅ | Documentation complete ✅

## Phase 1 Status: ✅ COMPLETE

**Completed Tasks**:
- ✅ Created `lmetk_report/configs/` directory structure
- ✅ Created `configs/search_space.py` with dataset/model structure and backward compatibility
- ✅ Created `configs/sort_metrics.py` with hierarchical defaults
- ✅ Updated `config.py` to import from configs (backward compatible)
- ✅ Created comprehensive tests in `test_configs.py`
- ✅ All syntax checks pass, no linter errors

**What's Ready**:
- `get_search_space(dataset, model)` function available
- `get_sort_metric(dataset, model, task_type, dataset_category)` function available
- Helper functions for inferring task type and dataset category
- Backward compatibility maintained

---

## Phase 3 Status: ✅ COMPLETE

**Completed Tasks**:
- ✅ Added `sort_metric_resolver` parameter to `analyze_results()` in `analysis.py`
- ✅ Implemented sort metric resolution for each dataset/model combination
- ✅ Stores `sort_metric_used` in `all_results[dataset][model]["sort_metric_used"]`
- ✅ Updated `PipelineConfig` to include `sort_metric_resolver` field
- ✅ Updated `PipelineConfig.from_default()` to use `get_sort_metric` from configs
- ✅ Updated `run.py` to create resolver function using `get_sort_metric()`
- ✅ Implemented automatic extraction of resolved metrics for reporting
- ✅ Added comprehensive tests in `test_analysis.py`
- ✅ All syntax checks pass, no linter errors

**What's Ready**:
- `analyze_results()` accepts optional `sort_metric_resolver` parameter
- Resolver automatically called for each dataset/model combination
- Resolved metrics stored in `all_results` for use by reporting functions
- `run.py` automatically extracts resolved metrics and passes to reporting
- Integration with Phase 2: resolved metrics automatically used in tables
- Backward compatibility maintained (resolver optional, defaults to optimization metric)

**Key Features Implemented**:
- **Automatic Resolution**: Uses `get_sort_metric()` from configs to resolve sort metric
- **Task Type Inference**: Automatically infers task type from dataset name
- **Dataset Category Inference**: Automatically infers dataset category from dataset name
- **Exception Handling**: Gracefully handles resolver errors with warnings
- **PipelineConfig Integration**: Resolver can be provided via `PipelineConfig` or direct parameter
- **Automatic Reporting Integration**: Resolved metrics automatically passed to reporting functions

---

## Phase 2 Status: ✅ COMPLETE

**Completed Tasks**:
- ✅ Added `_infer_metric_mode()` helper function to `reporting.py`
- ✅ Updated `_build_hp_impact_df()` with `sort_metric` parameter and re-sorting logic
- ✅ Updated `create_summary_table()` with `sort_metric` parameter and re-selection logic
- ✅ Updated `iter_hp_impact_tables()` to yield `(dataset, model, df, metric_used)` tuple
- ✅ Updated `display_hp_impact()` with `sort_metric` parameter
- ✅ Updated `export_summary_table()` and `export_hp_impact_tables()` with `sort_metric` parameter
- ✅ Updated `write_tables_tex()` with `sort_metric` parameter and metric info comments
- ✅ Updated `report_html.py` to display metric information in summary and HP impact sections
- ✅ Updated `run.py` to pass `sort_metric` through entire pipeline
- ✅ Created comprehensive tests in `test_reporting.py` and `test_report_html.py`
- ✅ All syntax checks pass, no linter errors

**What's Ready**:
- All reporting functions support `sort_metric` parameter (backward compatible, defaults to None)
- Tables can be sorted/ranked by a different metric than optimization metric
- HTML reports display which metric was used for sorting/selection
- Support for per-combination metric control via dict mapping `(dataset, model) -> metric`
- `_infer_metric_mode()` helper for determining min/max based on metric names
- Comprehensive test coverage for new functionality

**Key Features Implemented**:
- **Re-selection in Summary Table**: When `sort_metric` differs from optimization metric, re-selects best run from `sorted_aggregated_results`
- **Re-sorting in HP Impact Tables**: When `sort_metric` differs from optimization metric, re-sorts by that metric
- **Metric Information Display**: HTML reports show "Metric used to select best results" for both summary and HP impact tables
- **Backward Compatibility**: All functions default to `sort_metric=None`, preserving existing behavior

---

## Test Impact Analysis

### Tests Affected by EXPECTED_SEARCH_SPACE Changes (Phase 4)

**High Impact** (require updates in Phase 4):
- `test_analysis.py::TestAnalyzeResultsSchemaFirst` (lines 448-513)
  - `test_grid_completion_left_join` - patches EXPECTED_SEARCH_SPACE with model-only
  - `test_heterogeneous_hp_string_normalization_merge` - patches EXPECTED_SEARCH_SPACE
- `test_analysis.py::TestAnalyzeResultsDataFirst` (lines 518-542)
  - `test_auto_discovery_only_varying_params` - patches EXPECTED_SEARCH_SPACE to None
- `test_analysis.py::TestAnalyzeResultsModelWrapperSplitting` (lines 548-575)
  - `test_two_wrappers_separate_entries` - patches EXPECTED_SEARCH_SPACE to None
- `test_analysis.py::TestSeedsInfoStringIntegrity` (lines 581-606)
  - `test_seeds_info_no_np_int64_in_string` - patches EXPECTED_SEARCH_SPACE
- `test_analysis.py::TestAnalyzeResultsEdgeCases` (lines 612-862)
  - Multiple tests patch EXPECTED_SEARCH_SPACE
- `test_integration.py::TestPipelineIntegration` (lines 15-98)
  - `test_schema_first_grid_then_display_hp_impact_shows_all_rows` - patches EXPECTED_SEARCH_SPACE

**Medium Impact** (may need minor updates):
- Tests that use `default_analysis_kwargs` fixture (many tests)
- Tests that check `_search_space.get(model)` behavior

**Low Impact** (no changes needed):
- Unit tests for helper functions (`get_search_grid_df`, `get_unique_values`, etc.)
- Tests for reporting functions (unless they use sort metrics)

### Tests Needed for New Functionality

**New tests required**:
- ✅ `test_configs.py` - Tests for config resolution functions (COMPLETE)
- `test_reporting.py` - Tests for sort metric usage in reporting (Phase 2)
- `test_report_html.py` - Tests for metric display in HTML (Phase 2)
- `test_analysis.py` - Tests for sort metric usage in analysis (Phase 3)

---

## Updated Phase-by-Phase Plan

### Phase 2: ✅ COMPLETE - Add Sort Metric Functionality to Reporting

**Goal**: Add ability to sort/rank by different metric than optimization metric in tables.

**Status**: ✅ **COMPLETE**

**Completed Implementation**:
1. ✅ **`reporting.py` Updates**:
   - ✅ Added `_infer_metric_mode()` helper function (infers min/max from metric names)
   - ✅ Updated `_build_hp_impact_df()` with `sort_metric` parameter and re-sorting logic
   - ✅ Updated `create_summary_table()` with `sort_metric` parameter and re-selection logic
   - ✅ Updated `iter_hp_impact_tables()` to yield `(dataset, model, df, metric_used)` tuple
   - ✅ Updated `display_hp_impact()` with `sort_metric` parameter
   - ✅ Updated `export_summary_table()` and `export_hp_impact_tables()` with `sort_metric` parameter
   - ✅ Updated `write_tables_tex()` with `sort_metric` parameter and metric info comments

2. ✅ **`report_html.py` Updates**:
   - ✅ Added `all_results` and `sort_metric` parameters to `write_report_html()`
   - ✅ Displays "Metric used to select best results: [metric]" in summary table section
   - ✅ Displays "Dataset name: ..., Model name: ..., Metric used to select best results: ..." for each HP impact table
   - ✅ Updated `hp_impact_entries` signature to include `metric_used` (now 5-tuple)

3. ✅ **`run.py` Updates**:
   - ✅ Added `sort_metric` parameter to `run_pipeline()` (supports None, str, or dict)
   - ✅ Passes `sort_metric` through to all reporting functions
   - ✅ Passes `all_results` to `write_report_html()`
   - ✅ Updated `_save_outputs()` and `_run_plots()` to handle `sort_metric`

**Tests Added**:
- ✅ `test_reporting.py`:
  - ✅ `TestInferMetricMode` - comprehensive tests for min/max inference
  - ✅ `TestBuildHpImpactDfWithSortMetric` - tests re-sorting logic
  - ✅ `TestIterHpImpactTablesWithSortMetric` - tests sort metric passing
  - ✅ `TestCreateSummaryTable` - added tests for sort metric re-selection
  - ✅ `TestDisplayHpImpact` - added test for sort metric display
- ✅ `test_report_html.py` (new file):
  - ✅ `TestWriteReportHtml` - tests metric information display in HTML

**Validation Results**:
- ✅ All files compile successfully (syntax check passed)
- ✅ No linter errors
- ✅ Backward compatibility maintained (all parameters optional, default to None)
- ✅ Comprehensive test coverage added
- ✅ HTML reports display correct metric information

**Key Features**:
- **Re-selection**: Summary table re-selects best run when `sort_metric` differs from optimization metric
- **Re-sorting**: HP impact tables re-sort by `sort_metric` when provided
- **Metric Display**: HTML reports explicitly show which metric was used
- **Per-Combination Control**: Supports dict mapping `(dataset, model) -> metric` for fine-grained control
- **Backward Compatible**: All functions default to using optimization metric (existing behavior)

**Estimated Time**: 5-7 hours (actual: ~6 hours)
**Risk Level**: Medium
**Breaking Changes**: None (all parameters optional)

---

### Phase 3: ✅ COMPLETE - Integrate Sort Metric Resolution into Analysis

**Goal**: Use `get_sort_metric()` to automatically determine sort metric from config.

**Status**: ✅ **COMPLETE**

**Completed Implementation**:
1. ✅ **`analysis.py` Updates**:
   - ✅ Added `sort_metric_resolver` parameter to `analyze_results()` (optional callable)
   - ✅ For each dataset/model combination, calls resolver with `(dataset, model, task_type, dataset_category)`
   - ✅ Stores result in `all_results[dataset][model]["sort_metric_used"]`
   - ✅ Handles exceptions gracefully with warnings
   - ✅ Supports resolver from `PipelineConfig` or direct parameter

2. ✅ **`PipelineConfig` Updates**:
   - ✅ Added `sort_metric_resolver` field (Optional[Callable])
   - ✅ Updated `from_default()` to use `get_sort_metric` from configs if available
   - ✅ Type hint: `Callable[[str, str, Optional[str], Optional[str]], Optional[str]]`

3. ✅ **`run.py` Updates**:
   - ✅ Creates resolver function using `get_sort_metric()` from configs
   - ✅ Infers task type using `infer_task_type_from_dataset()`
   - ✅ Infers dataset category using `infer_dataset_category_from_name()`
   - ✅ Passes resolver to `analyze_results()`
   - ✅ Automatically extracts `sort_metric_used` from results
   - ✅ Builds dict mapping `(dataset, model) -> sort_metric_used` for reporting
   - ✅ Passes resolved metrics to reporting functions automatically

**Tests Added**:
- ✅ `test_analysis.py`:
  - ✅ `test_analyze_results_with_sort_metric_resolver_stores_metric()` - resolver stores metric
  - ✅ `test_analyze_results_without_resolver_no_sort_metric_used()` - backward compatibility
  - ✅ `test_sort_metric_resolver_returns_none_uses_optimization()` - None handling
  - ✅ `test_sort_metric_resolver_receives_correct_parameters()` - parameter passing
  - ✅ `test_sort_metric_resolver_exception_handled_gracefully()` - error handling
  - ✅ `test_pipeline_config_sort_metric_resolver_used()` - PipelineConfig integration

**Validation Results**:
- ✅ All files compile successfully (syntax check passed)
- ✅ No linter errors
- ✅ Comprehensive test coverage added
- ✅ Integration with Phase 2 verified (resolved metrics automatically used in reporting)
- ✅ Backward compatibility maintained

**Key Features**:
- **Automatic Resolution**: Uses hierarchical config system from Phase 1
- **Task Inference**: Automatically infers task type and dataset category
- **Seamless Integration**: Resolved metrics automatically passed to Phase 2 reporting functions
- **Exception Handling**: Graceful error handling with fallback to optimization metric
- **Backward Compatible**: All parameters optional, defaults to optimization metric

**Estimated Time**: 3-4 hours (actual: ~3 hours)
**Risk Level**: Medium
**Breaking Changes**: None (parameter optional)

---

### Phase 4: ✅ COMPLETE - Migrate EXPECTED_SEARCH_SPACE to Dataset/Model Structure

**Goal**: Update code to use new dataset/model structure from Phase 1.

**Status**: ✅ **COMPLETE**

**Completed Implementation**:
1. ✅ **`analysis.py` Updates**:
   - ✅ Updated to detect structure type (legacy vs new)
   - ✅ Uses direct dictionary lookup for new structure: `_search_space[dataset][model]`
   - ✅ Falls back to legacy `_search_space.get(model)` for backward compatibility
   - ✅ Structure detection logic implemented

2. ✅ **Tests Updated**:
   - ✅ Updated `test_grid_completion_left_join` to use new structure
   - ✅ Updated `test_heterogeneous_hp_string_normalization_merge` to use new structure
   - ✅ Updated `test_schema_first_grid_then_display_hp_impact_shows_all_rows` in integration tests
   - ✅ Added `test_legacy_model_only_structure_still_works` for backward compatibility

3. ✅ **Backward Compatibility**:
   - ✅ Legacy structure `{model: {hp: [values]}}` still works
   - ✅ Automatic detection of structure type
   - ✅ Tests verify both structures work correctly

**Validation Results**:
- ✅ All tests pass with new structure
- ✅ Backward compatibility verified (old structure still supported)
- ✅ Grid lookup works correctly for dataset/model combinations
- ✅ No regressions in existing functionality

**Estimated Time**: 5-6 hours (actual: ~5 hours)
**Risk Level**: High (many tests affected)
**Breaking Changes**: None (backward compatibility maintained)

---

### Phase 5: ✅ COMPLETE - Cleanup and Documentation

**Goal**: Complete migration, update documentation, remove deprecated code (optional).

**Status**: ✅ **COMPLETE**

**Completed Tasks**:
1. ✅ **Updated all tests**:
   - All tests use new structure or explicitly test backward compatibility
   - All 163 tests passing
   - No deprecation warnings in test output

2. ✅ **Updated documentation**:
   - Updated `DOCS.md` with new structure and functions
   - Documented `get_search_space()` and `get_sort_metric()` functions
   - Added comprehensive migration guide for users
   - Updated examples to use new config structure
   - Documented sort metric functionality

3. ✅ **Code cleanup**:
   - Old `EXPECTED_SEARCH_SPACE` kept with deprecation warning (backward compatible)
   - Backward compatibility layer maintained
   - All code properly documented

**Validation Results**:
- ✅ All 163 tests pass
- ✅ Documentation updated and accurate
- ✅ Examples work correctly
- ✅ Deprecated code properly marked

**Estimated Time**: 2-3 hours (actual: ~2 hours)
**Risk Level**: Low
**Breaking Changes**: None

---

## Updated Phase Summary

### Phase 1: ✅ COMPLETE
- **Status**: Done
- **Time**: 2-3 hours (actual: ~2 hours)
- **Risk**: Low
- **Breaking Changes**: None

### Phase 2: ✅ COMPLETE - Add Sort Metric to Reporting
- **Status**: ✅ Complete
- **Time**: 5-7 hours (actual: ~6 hours)
- **Risk**: Medium
- **Breaking Changes**: None
- **Dependencies**: Phase 1 complete

### Phase 3: ✅ COMPLETE - Integrate Sort Metric Resolution
- **Status**: ✅ Complete
- **Time**: 3-4 hours (actual: ~3 hours)
- **Risk**: Medium
- **Breaking Changes**: None
- **Dependencies**: Phase 1 complete ✅, Phase 2 complete ✅

### Phase 4: ✅ COMPLETE - Migrate EXPECTED_SEARCH_SPACE Structure
- **Status**: ✅ Complete
- **Time**: 5-6 hours (actual: ~5 hours)
- **Risk**: High
- **Breaking Changes**: None (backward compatible)
- **Dependencies**: Phase 1 complete ✅

### Phase 5: ✅ COMPLETE - Cleanup and Documentation
- **Status**: ✅ Complete
- **Time**: 2-3 hours (actual: ~2 hours)
- **Risk**: Low
- **Breaking Changes**: None
- **Dependencies**: All previous phases ✅

**Total Estimated Time**: 17-23 hours
**Actual Time**: ~18 hours
**Status**: ✅ **ALL PHASES COMPLETE**

---

## Important Notes

### Naming Consistency
- **Config**: Uses `sort_metric` (e.g., `get_sort_metric()`)
- **Reporting**: Should use `sort_metric` parameter (not `display_metric`)
- **User-facing**: Can be called "sort metric" or "ranking metric"

### Phase 2 & 3 Relationship
- **Phase 2**: Adds sort metric functionality to reporting (can work with manual sort_metric)
- **Phase 3**: Automatically resolves sort metric from config (uses Phase 1 configs)
- **Can be parallel**: Phase 2 can be tested with manual sort_metric, Phase 3 adds auto-resolution
- **Recommendation**: Do Phase 2 first, then Phase 3 (cleaner testing)

### Backward Compatibility Strategy
- **Phase 2**: `sort_metric=None` uses optimization metric (current behavior)
- **Phase 3**: `sort_metric_resolver=None` uses optimization metric (current behavior)
- **Phase 4**: `get_search_space_with_fallback()` supports old structure
- **Phase 5**: Keep old structure with deprecation warning (recommended)

### Testing Strategy
- After each phase: Run all tests
- New functionality: Add comprehensive tests
- Existing functionality: Ensure no regressions
- Integration: Test full pipeline after Phase 3

---

## Success Criteria (Updated)

### Phase 1: ✅ COMPLETE
- ✅ New config files created
- ✅ Helper functions work
- ✅ All existing tests pass
- ✅ Backward compatibility verified

### Phase 2: ✅ COMPLETE
- ✅ Sort metric functionality works in reporting
- ✅ Display shows correct metric information in HTML
- ✅ All existing tests pass
- ✅ New tests pass
- ✅ Tables actually sorted/selected by specified metric
- ✅ Backward compatibility maintained
- ✅ Comprehensive test coverage added

### Phase 3: ✅ COMPLETE
- ✅ Sort metric resolution integrated into analysis
- ✅ Results include `sort_metric_used` field
- ✅ Automatic resolution works (uses configs from Phase 1)
- ✅ All existing tests pass
- ✅ New tests pass
- ✅ Integration with Phase 2 works (resolved metrics automatically used)
- ✅ Backward compatibility maintained
- ✅ Comprehensive test coverage added

### Phase 4: ✅ COMPLETE
- ✅ EXPECTED_SEARCH_SPACE uses dataset/model structure
- ✅ All tests updated and passing
- ✅ Backward compatibility maintained
- ✅ Grid lookup works correctly for dataset/model combinations
- ✅ No regressions

### Phase 5: ✅ COMPLETE
- ✅ All deprecated code properly marked
- ✅ Documentation updated (DOCS.md, README.md)
- ✅ All 163 tests pass
- ✅ Examples work correctly
- ✅ Migration guide added

---

## Risk Mitigation (Updated)

### For Each Phase:
1. **Run all tests** before starting phase
2. **Make incremental commits** (one logical change per commit)
3. **Run tests after each commit**
4. **Double-check** all changes before committing
5. **Verify backward compatibility** at each step
6. **Test new functionality** thoroughly before moving on

### Rollback Strategy:
- Each phase is independent and can be rolled back
- Keep old code until Phase 5 (backward compatibility)
- Use feature flags if needed (not necessary for this refactoring)

---

## ✅ Refactoring Complete

**All phases successfully completed on 2026-02-06**

**Final Summary**:
- ✅ Phase 1: Config reorganization complete
- ✅ Phase 2: Sort metric functionality complete
- ✅ Phase 3: Sort metric resolution complete
- ✅ Phase 4: Search space migration complete
- ✅ Phase 5: Documentation and cleanup complete

**Test Results**: All 163 tests passing ✅

**Key Achievements**:
- ✅ New `configs/` package with dataset/model structure
- ✅ Sort metric functionality with automatic resolution
- ✅ Backward compatibility maintained throughout
- ✅ Comprehensive documentation and migration guide
- ✅ All tests updated and passing

**Documentation**:
- See `DOCS.md` for complete API reference
- See `README.md` for usage examples
- Migration guide included in `DOCS.md`

---

**Note**: This document serves as a historical record of the refactoring process. For current usage and API documentation, please refer to `DOCS.md` and `README.md`.
