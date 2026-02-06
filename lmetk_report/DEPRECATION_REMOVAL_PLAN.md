# Deprecation Removal and Test Coverage Plan

**Goal**: 
1. Achieve 100% test coverage for `lmetk_report/` (excluding `not_used/` and `plots.py`)
2. Remove deprecated code in phases, ensuring tests pass after each phase
3. Remove `plots.py` entirely (not needed)
4. Consider moving `config.py` to `configs/` directory

**Status**: Test coverage gaps addressed (Phase 0 in REFACTORING_PLAN_V2). Current: 194 tests, ~81% coverage; test_preprocess, test_utils, test_run, test_config in place. Deprecation/removal phases remain optional; see REFACTORING_PLAN_V2 Phase 4.

---

## Part 1: Test Coverage Analysis

### Current Test Coverage Status

#### ✅ Well Covered Modules:
- **`analysis.py`** - `test_analysis.py` (comprehensive tests)
- **`reporting.py`** - `test_reporting.py` (comprehensive tests)
- **`validators.py`** - `test_validators.py` (comprehensive tests)
- **`run_processor.py`** - `test_run_processor.py` (comprehensive tests)
- **`configs/`** - `test_configs.py` (comprehensive tests)
- **`report_html.py`** - `test_report_html.py` (comprehensive tests)
- **Integration** - `test_integration.py` (end-to-end tests)

#### ⚠️ Potentially Missing Coverage:
- **`preprocess.py`** - No dedicated test file found
  - `clean_and_rename_models()` - needs tests
  
- **`utils.py`** - No dedicated test file found
  - `format_mean_std_count()` - needs tests
  - `flatten_aggregated_columns()` - needs tests (may be tested indirectly)
  
- **`run.py`** - Partial coverage via integration tests
  - `run_pipeline()` - tested via integration
  - `_safe_basename()` - needs explicit tests
  - `_save_outputs()` - needs explicit tests
  - `_run_plots()` - needs explicit tests (but will be removed if plots.py is removed)
  - `main()` - needs explicit tests

- **`config.py`** - Partial coverage
  - `PipelineConfig` class - needs comprehensive tests
  - `PipelineConfig.from_default()` - needs tests

#### ❌ To Be Removed:
- **`plots.py`** - Will be removed entirely (not needed for core functionality)

### Test Coverage Gaps to Address

1. **`test_preprocess.py`** (NEW) - Test preprocessing functions
2. **`test_utils.py`** (NEW) - Test utility functions
3. **`test_run.py`** (NEW) - Test run.py functions (or extend integration tests)
4. **`test_config.py`** (NEW or extend) - Test PipelineConfig thoroughly

---

## Part 2: Deprecated Code Identification

### Deprecated Code to Remove:

#### 1. **`config.py`** - Legacy `EXPECTED_SEARCH_SPACE`
- **Location**: Lines 129-247
- **Status**: ⚠️ DEPRECATED - marked with deprecation comment
- **Replacement**: Use `lmetk_report.configs.search_space.EXPECTED_SEARCH_SPACE`
- **Impact**: 
  - Used as fallback in `analysis.py` (line 309)
  - Referenced in `PipelineConfig.from_default()` (line 235-237)
  - Used in tests (multiple test files patch it)

#### 2. **`analysis.py`** - Legacy structure detection and fallback
- **Location**: 
  - Lines 315-330: Legacy structure detection logic (`_is_legacy_search_space`)
  - Lines 489-497: Legacy lookup fallback (direct dict access vs `_search_space.get(model)`)
  - Lines 22-30: Import fallback with nested `get_search_space_with_fallback` function (unused - never called)
- **Status**: Backward compatibility code
- **Impact**: 
  - Tests rely on legacy structure (`test_legacy_model_only_structure_still_works`)
  - Multiple tests patch `EXPECTED_SEARCH_SPACE` with legacy structure
- **Note**: When `model_grid` is `None` (search space not found), the system correctly falls back to auto-discovery (data-first mode). This behavior should be preserved.

#### 3. **`configs/search_space.py`** - Legacy conversion helpers
- **Location**:
  - `_convert_legacy_search_space()` - Lines 95-128 (NOT USED - can be removed)
  - `get_search_space_with_fallback()` - Lines 131-162 (NOT USED - imported but never called)
- **Status**: Backward compatibility helpers that are not actually used
- **Impact**: 
  - `get_search_space_with_fallback` is imported in `analysis.py` but never called
  - The actual fallback logic is in `analysis.py` lines 489-497 (direct dict access)
  - When search space is not found, system correctly uses auto-discovery (varying HPs)
- **Clarification**: The "fallback" the user wants to keep is NOT to legacy structure, but to **auto-discovery** when search space is not configured. This is already working correctly.

#### 4. **`plots.py`** - To be removed entirely
- **Location**: Entire file
- **Status**: Not needed for core functionality
- **Impact**: 
  - Used in `run.py` (`_run_plots()` function)
  - Exported in `__init__.py`
  - Need to remove from exports and update `run.py`

#### 5. **`config.py` location** - Consider moving to `configs/`
- **Current**: `lmetk_report/config.py`
- **Proposed**: `lmetk_report/configs/config.py` (or keep as is)
- **Analysis**:
  - Contains core configs: `COLUMN_CONFIG`, `COLUMNS_TO_NORMALIZE`, `SPECIAL_COLUMNS`, etc.
  - Contains deprecated `EXPECTED_SEARCH_SPACE` (to be removed in Phase 4)
  - Contains `PipelineConfig` dataclass
  - Imported widely: 
    - `from lmetk_report import config` (6 files: analysis.py, __init__.py, preprocess.py, run_processor.py, conftest.py)
    - `from lmetk_report.config import PipelineConfig` (4 files: analysis.py, __init__.py, run_processor.py, test_analysis.py)
- **Impact**: 
  - Moving would require updating 10+ import statements
  - Would break external code that imports `lmetk_report.config`
  - `configs/` directory currently contains dataset/model specific configs (search_space, sort_metrics)
  - `config.py` contains general pipeline configuration (columns, filters, etc.)
- **Recommendation**: 
  - **Keep as `lmetk_report/config.py`** for now (less disruptive, clearer separation)
  - `configs/` is for dataset/model-specific configurations
  - `config.py` is for general pipeline configuration
  - Can reconsider after deprecation removal if organization becomes clearer

#### 6. **Documentation references**
- **Location**: `DOCS.md`, `README.md`, `REFACTORING_PLAN.md`
- **Status**: References to deprecated patterns
- **Impact**: Documentation needs updates

---

## Part 3: Phased Removal Plan

### Phase 0: Achieve 100% Test Coverage ⚠️ CRITICAL FIRST STEP

**Goal**: Ensure all code is tested before removing deprecated code

**Tasks**:
1. ✅ Create `test_preprocess.py` with tests for:
   - `clean_and_rename_models()` - test prefix removal, model type appending
   - Edge cases: empty df, no common prefix, missing columns
   
2. ✅ Create `test_utils.py` with tests for:
   - `format_mean_std_count()` - test various inputs, edge cases (NaN, strings)
   - `flatten_aggregated_columns()` - test MultiIndex flattening, plain columns
   
3. ✅ Create `test_run.py` or extend integration tests with:
   - `_safe_basename()` - test filename sanitization
   - `_save_outputs()` - test file writing (mock filesystem, but skip plot generation)
   - `main()` - test CLI argument parsing
   - Note: `_run_plots()` will be removed when `plots.py` is removed
   
4. ✅ Extend `test_configs.py` or create `test_config.py` with:
   - `PipelineConfig` initialization
   - `PipelineConfig.from_default()` - test all fields
   - Edge cases and validation

**Validation**: Run coverage tool to verify 100% coverage (excluding `plots.py` which will be removed)
**Estimated Time**: 3-5 hours (reduced since plots.py tests not needed)
**Risk Level**: Low (adding tests only)

---

### Phase 1: Remove Unused Legacy Code (Low Risk)

**Goal**: Remove unused legacy code that's not actually called

**Tasks**:
1. Remove unused import fallback in `analysis.py` (lines 22-30):
   - Remove `get_search_space_with_fallback` from import
   - Remove nested function definitions (lines 25-30) - these are never called
   - Keep only `get_search_space` import
   
2. Remove unused functions from `configs/search_space.py`:
   - Remove `_convert_legacy_search_space()` (lines 95-128) - never used
   - Remove `get_search_space_with_fallback()` (lines 131-162) - imported but never called
   - Update `configs/__init__.py` if it exports these
   
3. Verify auto-discovery fallback still works:
   - When `get_search_space(dataset, model)` returns `None`, `model_grid` is `None`
   - System correctly falls back to data-first mode (auto-discovery of varying HPs)
   - This is the desired behavior and should be preserved

**Dependencies**: Phase 0 complete
**Validation**: All tests pass, verify auto-discovery still works
**Estimated Time**: 1-2 hours
**Risk Level**: Low (removing unused code only)

---

### Phase 2: Remove Legacy Structure Detection (Medium Risk)

**Goal**: Remove legacy structure detection logic from `analysis.py`

**Tasks**:
1. Remove legacy structure detection (lines 315-330):
   - Remove `_is_legacy_search_space` detection logic
   - Remove the check that determines if structure is legacy vs new
   
2. Simplify lookup logic (lines 489-497):
   - Remove `if _is_legacy_search_space:` branch
   - Keep only new structure lookup: `_search_space[dataset][model]`
   - When not found, `model_grid` is `None` → auto-discovery (data-first mode)
   
3. Update tests that rely on legacy structure:
   - Remove `test_legacy_model_only_structure_still_works` or convert to test new structure
   - Update tests that patch `EXPECTED_SEARCH_SPACE` with legacy structure to use new structure
   - Verify tests still pass with new structure only

**Dependencies**: Phase 1 complete
**Validation**: All tests pass, verify auto-discovery fallback still works
**Estimated Time**: 2-3 hours
**Risk Level**: Medium (tests may need updates)

---

### Phase 3: Remove `plots.py` (Low Risk)

**Goal**: Remove plotting functionality entirely

**Tasks**:
1. Remove `plots.py` file
2. Remove from `__init__.py` exports:
   - Remove `plot_best_metric_bars`, `plot_hp_impact` from `__all__`
   - Remove imports from `plots.py`
3. Update `run.py`:
   - Remove `_run_plots()` function or make it a no-op
   - Remove `show_plots` parameter from `run_pipeline()` or make it ignored
   - Remove plot-related imports
4. Update documentation:
   - Remove references to plotting functions
   - Update examples that use plots

**Dependencies**: Phase 0 complete (tests written, but plots.py not tested)
**Validation**: All tests pass, verify no imports break
**Estimated Time**: 1-2 hours
**Risk Level**: Low (plotting is optional functionality)

---

### Phase 4: Remove Deprecated `EXPECTED_SEARCH_SPACE` from `config.py` (High Risk)

**Goal**: Remove the deprecated `EXPECTED_SEARCH_SPACE` dictionary from `config.py`

**Tasks**:
1. Remove `EXPECTED_SEARCH_SPACE` from `config.py` (lines 129-247)
2. Update `analysis.py` to not fallback to `config.EXPECTED_SEARCH_SPACE` (line 309)
3. Update `PipelineConfig.from_default()` to not use `config.EXPECTED_SEARCH_SPACE` (lines 235-237)
4. Update all tests that patch `config.EXPECTED_SEARCH_SPACE`:
   - Find all `@patch('lmetk_report.config.EXPECTED_SEARCH_SPACE')` or similar
   - Update to patch `lmetk_report.configs.search_space.EXPECTED_SEARCH_SPACE` instead
   - Or update tests to use `PipelineConfig` with `expected_search_space` parameter
5. Update documentation:
   - Remove references to deprecated `EXPECTED_SEARCH_SPACE` in `DOCS.md`
   - Update `README.md` examples
   - Update migration guide

**Dependencies**: Phase 2 complete
**Validation**: All tests pass, verify no external code relies on `config.EXPECTED_SEARCH_SPACE`
**Estimated Time**: 4-6 hours
**Risk Level**: High (many tests affected, potential external dependencies)

---

### Phase 5: Clean Up Documentation and Final Verification (Low Risk)

**Goal**: Update all documentation and verify everything works

**Tasks**:
1. Update `DOCS.md`:
   - Remove deprecated structure examples
   - Update all references to use new structure
   - Update migration guide
2. Update `README.md`:
   - Remove deprecated examples
   - Update to use new config structure
3. Update `REFACTORING_PLAN.md`:
   - Mark deprecation removal as complete
   - Add notes about breaking changes (if any)
4. Run full test suite
5. Verify integration tests pass
6. Check for any remaining references to deprecated code

**Dependencies**: Phase 4 complete
**Validation**: All tests pass, documentation accurate
**Estimated Time**: 2-3 hours
**Risk Level**: Low

---

## Summary

### Total Estimated Time: 12-19 hours

### Risk Assessment:
- **Phase 0**: Low risk (adding tests)
- **Phase 1**: Low risk (removing unused code)
- **Phase 2**: Medium risk (removing detection logic)
- **Phase 3**: Low risk (removing optional plotting)
- **Phase 4**: High risk (removing main deprecated code)
- **Phase 5**: Low risk (documentation)

### Key Clarifications:

1. **Auto-Discovery Fallback**: When search space is not found in config (`model_grid` is `None`), the system correctly falls back to **auto-discovery** (data-first mode) where it finds varying hyperparameters automatically. This is the desired behavior and should be preserved.

2. **Legacy Structure Fallback**: The current code has fallback to legacy model-only structure, but this is NOT the same as auto-discovery. We want to remove legacy structure support, but keep auto-discovery.

3. **Unused Code**: `get_search_space_with_fallback()` and `_convert_legacy_search_space()` are imported/defined but never actually called. The actual fallback logic is direct dictionary access in `analysis.py`.

4. **Plots Removal**: `plots.py` is optional functionality and can be removed entirely. This simplifies the codebase.

### Critical Dependencies:
- Phase 0 MUST be completed first (100% test coverage)
- Each phase should be validated with full test suite before proceeding
- Get feedback from failing tests after each phase

### Breaking Changes:
- **Phase 3** may break external code that imports `config.EXPECTED_SEARCH_SPACE`
- Need to document migration path for external users
- Consider deprecation warnings before removal (if external users exist)

---

## Next Steps

1. ✅ Review this plan
2. ✅ Start with Phase 0 (test coverage)
3. ✅ Run coverage tool to verify gaps
4. ✅ Create missing test files
5. ✅ Proceed with phases sequentially, getting test feedback after each
