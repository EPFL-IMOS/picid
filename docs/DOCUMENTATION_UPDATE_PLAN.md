# Documentation Update Plan (status)

This document records what was updated to bring the docs in line with the library, and what was **intentionally left unchanged**.

---

## Completed updates (production pass)

The following were updated for a **professional, production-ready** documentation set:

| Item | Changes |
|------|--------|
| **mkdocs.yml** | Added full `nav` (Home, Getting started, Data, Transforms, Evaluators, Models & configuration, Running, API Reference, Archive). Added theme features and site metadata. |
| **index.md** | Replaced default MkDocs template with a proper homepage: overview of TabPHM, core concepts, documentation map table, and quick start. |
| **api.md** | Expanded from a single module to full data-objects API: `base` (mixins, BaseDataObject, BaseDataObjectWithMetadata), `data` (NamedTransformInput, DatasetContainer, SplitDatasetContainer, return types), and a utilities table for `utils.py`. Added “Related documentation” section. |
| **dataobject.md** | Added “Module location and API” (section 5) with file locations and link to API Reference. |
| **datasources.md** | Reworked with a complete summary table (all datasources, module column, domain, tasks, features). Added missing entries: phmd_cbmv3, phmd_hsf15, phmd_phme20, MZVAV, picasso, kubric_movie; aligned names (nb14, unibo). Added short descriptions for each datasource. Clarified intro (datasource vs dataset). |
| **datasets.md** | Tightened intro and terminology; added “See also” (datasets_inner_workings, dataobject, datasources) and module location (`lmetk.data.datasets`). |
| **datasets_inner_workings.md** | Added “Module locations” (datasets in `lmetk.data.datasets`, sequencers/engine in `lmetk.data.optimization`). Noted that the engine class name is `AkwardJaggedAutoregressiveSequencer` (historical “Akward” spelling). Added “See also” at the end. |
| **model_types.md** | Full rewrite: intro, three supported types (fit-predict, batchwise, full-batch), data shapes, foundation-style forecaster limitations, rolling vs static context, and “Where implementations live” with pointers to wrappers, baselines, and definitions. |
| **how_to_add_a_new_model.md** | Full rewrite: step-by-step (choose base class, implement wrapper, create configs, register, run). Documented AbstractFitPredictWrapper vs AbstractFeedForwardWrapper vs AbstractFeedForwardTrainingWrapper, config layout (model, model_config, experiment), and example wrappers table. |
| **how_to_use_configs.md** | Complete list of OmegaConf resolvers (flat, uuid, sum, prod, mod, int_div, quot, diff, infer_data_dim, infer_dataloader_length) with signatures and behaviour. Documented where they are registered (`lmetk/run.py`), runtime replacement of infer_* resolvers, and example YAML usage. |
| **viewer.md** | Short description of the dataset viewer, entry point, and start command with option notes. |
| **runai.md** | Added one-line “See also” link to Setup for local runs. Left body unchanged. |
| **old_setup.md** | Added legacy notice at the top pointing to [RunAI](runai.md) for the current workflow. Placed under **Archive** in nav. |

---

## Intentionally not modified

Per request, the following were **not** changed:

- **setup.md** — Left as-is.
- **docs/transforms** (all pages) and **docs/transforms.md** — Considered already in good shape; no edits.
- **docs/evaluators** (evaluators/index.md) — Considered already in good shape; no edits.

---

## Optional follow-ups (not done in this pass)

- Add more API reference pages (e.g. `lmetk.evaluator`, `lmetk.metrics`, `lmetk.pipeline`) if desired.
- Resolve or merge duplicate transform intros (`intro_5min` etc.) if they are still present and desired to be canonical.
- Keep the plan file as a living checklist for future doc updates.
