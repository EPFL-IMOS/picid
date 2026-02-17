# Transform Refactoring — Status Outline

**Sources:** [REFACTORING_ROADMAP.md](REFACTORING_ROADMAP.md), [TRANSFORM_ARCHITECTURE_AUDIT.md](TRANSFORM_ARCHITECTURE_AUDIT.md).

---

## Done ✅

| Phase | Delivered |
|-------|-----------|
| **1** | Stabilization suite; transform base tests green. |
| **2** | Registry-based dispatching: `DataKind`, `TransformCapability`, `infer_data_kind`, `get_capability`; handler registry; `DenseDenseHandler`, `RaggedDenseHandler`, `RaggedRaggedHandler`; strategy and multisource use registry. |
| **3** | Composable pipeline: `TransformContext`; `CopyStep`, `FitStep`, `TransformStep`, `PostprocessStep`, `MergeStep`, `RecordManifestStep`, `IntegrityCheckStep`; strategy runs step pipeline; lifecycle hooks. |
| **4.1** | Metadata manifest: schema, `MetadataManifest`, optional on container; `RecordManifestStep`; tests. |
| **4.2** | Slice-awareness: `SliceInfo`, optional on container; `TransformContext.slice_info`; tests. |
| **4.3** | Integrity: `IntegrityPolicy`, `run_check`/`run_checks`, `IntegrityCheckStep`, `context.integrity_checks`; doc on data-object vs pipeline checks. |
| **5.1** | Cache boundaries: `metadata.cache_point`; manager APIs; PreProcessor boundary restore; FileSystemCache `"boundary"` stage; `save_boundary_caches`; tests. |
| **5.2** | Structural dry run: `dry_run_transforms(manager, container)`; apply_to/assign_to and pipeline-produced keys; handler check; tests. |
| **5.3** | Inverse routing: `get_inverter_for_key(key, which)`; run.py `inverse_transform_key`; INVERSE_TRANSFORM_ROUTING.md. |
| **Tests** | Caching: `test_file_system_cache.py` (boundary), `test_preprocessor_caching.py`, `run_caching_coverage.sh`. |

---

## Not Done / Follow-ups

| Area | What’s left |
|------|-------------|
| **4.3** | Built-in pipeline check implementations (e.g. NaN in container keys); currently only runner + policy, callables injected by caller. |
| **4.3** | Per-transform `metadata.validation_policy` (expectations/overrides) in config; scoped overrides with reason and audit (audit §2.3.4). |
| **5.1** | Profiling: cache hit/miss per step, restore time, time saved. |
| **5.3** | Config example and resolution logging for `inverse_transform_key`. |
| **Audit** | Protocols and typed metadata (SupportsDense/SupportsRagged, `TransformMetadata` TypedDict); new data type (e.g. PointClouds) still requires core edits. |
| **Audit** | StepRecord / Execution Report: per-step identity, timing, cache info; end-of-run summary. |
| **Audit** | Validation-first execution: run invariants before and after *each* transform (current: single IntegrityCheckStep after CopyStep). |
| **Audit** | Lifecycle hook registry: confirm `on_pipeline_start`, `before_transform`, `after_transform`, `on_transform_error`, `on_pipeline_end` emitted on all paths (including cache hit). |

---

## Reference

- **Roadmap (phases, gates, details):** [REFACTORING_ROADMAP.md](REFACTORING_ROADMAP.md)  
- **Audit (friction, target state, wish list):** [TRANSFORM_ARCHITECTURE_AUDIT.md](TRANSFORM_ARCHITECTURE_AUDIT.md)  
- **Inverse routing usage:** [INVERSE_TRANSFORM_ROUTING.md](INVERSE_TRANSFORM_ROUTING.md)
