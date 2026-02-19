# Plan: Documentation refactor for webpage and current library state

**Goal:** Update TabPHM documentation so it (1) reflects the current state of the library, (2) is structured for a story-like webpage, (3) improves findability via subfolders and clear narrative, and (4) anchors learning on a concrete PHM example (e.g. Unibo battery/RUL) showing config groups and overrides. The piCID branding (logo) should be integrated where appropriate. The refactor is phased; later phases can refine without blocking an initial usable site.

**Architecture (doc tree):**

- **Entry:** `docs/index.md` — Keep as main landing; add piCID logo, one-sentence tagline, and clear “Start here” / “Example” / “Reference” sections.
- **Getting started:** `docs/getting-started/` — Setup, paths, first run, viewer (move/adapt from current `setup.md`, `viewer.md`).
- **Concepts (story order):** `docs/concepts/` — Data flow: datasource → data objects → datasets → transforms → model → evaluator → pipeline; one file per concept or a single “story” file with sections.
- **Guides:** `docs/guides/` — How-to: configs and overrides, adding a model, OmegaConf resolvers, RunAI (from `how_to_use_configs.md`, `how_to_add_a_new_model.md`, `runai.md`).
- **Example (end-to-end):** `docs/examples/` — One primary example: Unibo battery/RUL (datasource, task_definition, transforms: raw vs combined vs ablation/missing_values), run commands and config snippets; optional second example (e.g. PHM20 or railway) later.
- **Reference:** `docs/reference/` — Datasources table, datasets, transforms (index + link to existing `transforms/`), evaluators, model types, API (from `datasources.md`, `datasets.md`, `transforms.md`, `evaluators/`, `model_types.md`, `api.md`).
- **Plans:** `docs/plans/` — This file and future planning docs (e.g. phase checklists).

**Tech stack:** Markdown docs, static-site friendly (MkDocs or similar later); no code changes required for Phase 1–4 beyond doc content and file moves.

---

## Phases (overview)

| Phase | Focus | Outcome |
|-------|--------|---------|
| **1** | Structure and index | New folder layout, `index.md` updated, redirects/links from old paths. |
| **2** | Story and concepts | Single “From data to prediction” narrative; concepts in story order. |
| **3** | Config and overrides | Guide: config groups, defaults, `override /transforms`, experiment variants. |
| **4** | Unibo battery example | End-to-end example: datasource, task, transforms (raw/combined/ablation), run commands. |
| **5** | Reference and deep dives | Datasources, datasets, transforms, evaluators, model types, API in reference layout. |
| **6** | Webpage polish | Logo/branding, navigation, optional MkDocs/site config, link checks. |

---

## Phase 1: Structure and index

**Objective:** Introduce new doc structure and update the main entry so the site can be read top-to-bottom and by role (new user vs reference).

1. **Create directories** (no content yet):  
   - `docs/getting-started/`  
   - `docs/concepts/`  
   - `docs/guides/`  
   - `docs/examples/`  
   - `docs/reference/`  
   - `docs/plans/` (this file lives here).

2. **Update `docs/index.md`:**  
   - Add a short “About” line and **piCID logo** (image path, e.g. `assets/...` or link to existing logo asset).  
   - Add three clear sections: **Start here** (link to `getting-started/`), **Learn the framework** (link to `concepts/` and `examples/`), **Reference** (link to `reference/`).  
   - Keep a minimal “Documentation map” table pointing to new locations (getting-started, concepts, guides, examples, reference).  
   - Keep Quick start steps but optionally link “Setup” to `getting-started/setup.md`.

3. **Create stub index files** so navigation is obvious:  
   - `docs/getting-started/index.md` — Short intro + list of pages (Setup, Paths, First run, Viewer).  
   - `docs/concepts/index.md` — Short intro + list of concept pages or “one story” page.  
   - `docs/guides/index.md` — Short intro + list of guides (Configs, Add model, Resolvers, RunAI).  
   - `docs/examples/index.md` — Short intro + “Unibo RUL example” (and placeholder for more).  
   - `docs/reference/index.md` — Short intro + list of reference pages (Datasources, Datasets, Transforms, Evaluators, Model types, API).

4. **Move and relink (bite-sized):**  
   - Move `docs/setup.md` → `docs/getting-started/setup.md`; fix any in-repo links that pointed to `setup.md`.  
   - Move `docs/viewer.md` → `docs/getting-started/viewer.md`; fix links.  
   - In `docs/index.md` and any other doc, replace links to `setup.md` / `viewer.md` with `getting-started/setup.md` / `getting-started/viewer.md`.

5. **Optional:** Add a one-line note in `README.md` that full docs live under `docs/` and link to `docs/index.md`.

**Exit criterion:** New folders exist, `index.md` has logo + three sections + doc map, stub indexes exist, setup/viewer moved and linked.

---

## Phase 2: Story and concepts

**Objective:** Provide one narrative (“From data to prediction”) and place concepts in reading order.

1. **Create the story page** `docs/concepts/from_data_to_prediction.md`:  
   - Title: e.g. “From data to prediction: how one PHM run works.”  
   - Sections in order: (1) Datasource loads raw data into containers, (2) Data objects hold splits and metadata, (3) Datasets (e.g. RUL context, fit-predict) build model inputs, (4) Transforms preprocess (fit-on-train, keys), (5) Model and training, (6) Evaluators and metrics, (7) Pipeline ties it together (Hydra, config).  
   - Use the Unibo RUL setup as a running example (names only in this phase: datasource unibo21_bosello, task prognostics/rul, transforms raw/combined/ablation).  
   - Link to detailed reference pages for each concept.

2. **Create or split concept pages** under `docs/concepts/`:  
   - Either one file per concept: `data_flow.md`, `datasources.md`, `data_objects.md`, `datasets.md`, `transforms.md`, `evaluators.md`, `pipeline.md` (content can be short and point to reference),  
   - Or keep a single `concepts/from_data_to_prediction.md` and add a short `concepts/index.md` that lists “Data flow” … “Pipeline” as subsections of that story (no obligation to split into many files in phase 2).

3. **Sync with existing docs:**  
   - Ensure `docs/concepts/` does not duplicate `docs/dataobject.md`, `docs/datasets.md`, `docs/transforms.md` verbatim; either move those into `docs/reference/` in Phase 5 or have concepts summarize and link to them.  
   - Update `docs/index.md` “Learn the framework” to link to `concepts/from_data_to_prediction.md` and `concepts/index.md`.

**Exit criterion:** One story document exists and is linked from index; concepts are in story order; no broken links.

---

## Phase 3: Config and overrides guide

**Objective:** Explain how experiments are composed from config groups and overrides (so the Unibo example in Phase 4 is understandable).

1. **Create `docs/guides/configs_and_overrides.md`:**  
   - Describe **experiment** as a composition: `experiment=<domain>/<task>/<variant>/<model>` (e.g. `unibo/prognostics/raw/cnn_1d`).  
   - Explain **defaults** and **override**: e.g. `override /datasource: ...`, `override /task_definition: ...`, `override /transforms: ...`.  
   - Show a minimal example: one experiment YAML with `defaults` and one override for `/transforms`.  
   - Document **paths** and **debug** as common CLI overrides.  
   - Link to `how_to_use_configs.md` (resolvers) and to the Unibo example (Phase 4).

2. **Relocate resolvers content:**  
   - Move `docs/how_to_use_configs.md` → `docs/guides/omegaconf_resolvers.md` (or keep in place and add a link from `guides/configs_and_overrides.md`).  
   - In `docs/guides/index.md`, add “Config and overrides” and “OmegaConf resolvers”.

3. **Optional:** Add a small diagram (Mermaid or list) of config composition: experiment → datasource, task_definition, dataset, transforms, model, evaluator.

**Exit criterion:** Config/overrides guide exists and is linked from guides index; resolvers doc linked; Unibo example referenced.

---

## Phase 4: Unibo battery RUL example

**Objective:** One end-to-end, copy-paste-friendly example that shows framework usage via configs.

1. **Create `docs/examples/unibo_rul.md`:**  
   - **Title:** e.g. “Unibo battery RUL: from data to prediction.”  
   - **What you’ll run:** `python lmetk/run.py paths=<paths> experiment=unibo/prognostics/raw/cnn_1d` (and optionally `debug=default`).  
   - **Config layout:**  
     - `configs/experiment/unibo/base.yaml` (datasource, evaluator, task_definition).  
     - `configs/experiment/unibo/prognostics/base.yaml` (per_unit evaluator, prognostics/rul, dataset with `get_unit_id`).  
     - `configs/experiment/unibo/prognostics/raw/cnn_1d.yaml`: defaults + `override /transforms: battery/unibo/raw`.  
   - **Variants:**  
     - Same model, different preprocessing: `experiment=unibo/prognostics/combined/cnn_1d` (combined transforms).  
     - Ablation: `experiment=unibo/prognostics/ablation/missing_values/cnn_1d` (ablation_missing_values_combined).  
   - **Snippets:** Paste minimal YAML for one experiment and one override; explain what changes when switching transforms.  
   - **Datasource:** Point to `configs/datasource/unibo21_bosello.yaml` and task_definition `prognostics/rul` (seq_len, stride, etc.).  
   - Link to concepts (data flow) and to config guide (overrides).

2. **Update `docs/examples/index.md`:**  
   - List “Unibo RUL” as the main example and briefly describe it (battery/RUL, config groups, transform overrides).

3. **Optional:** Add a “Next steps” (e.g. try PHM20 or railway experiment, or change model in the same Unibo tree).

**Exit criterion:** Unibo RUL example doc exists with run commands and config references; linked from examples index and from config guide.

---

## Phase 5: Reference and deep dives

**Objective:** Organize existing reference material into `docs/reference/` and keep deep-dive material (e.g. transforms) easy to find.

1. **Move reference pages:**  
   - `docs/datasources.md` → `docs/reference/datasources.md`  
   - `docs/datasets.md` → `docs/reference/datasets.md`  
   - `docs/datasets_inner_workings.md` → `docs/reference/datasets_inner_workings.md`  
   - `docs/transforms.md` → `docs/reference/transforms.md` (and keep `docs/transforms/` as subfolder or move under `reference/transforms/` — decide one place for “transform quick start / guide / developer guide”).  
   - `docs/model_types.md` → `docs/reference/model_types.md`  
   - `docs/api.md` → `docs/reference/api.md`  
   - Evaluators: keep `docs/evaluators/index.md` or move to `docs/reference/evaluators.md` and move evaluators content there.  
   - Fix all in-repo links that pointed to the old paths.

2. **Transforms:**  
   - Either leave `docs/transforms/` at top level and link from `docs/reference/transforms.md`, or move `docs/transforms/*` into `docs/reference/transforms/` and update links.  
   - Ensure “Quick start / User guide / Developer guide” remain linked from the main transforms page.

3. **Reference index:**  
   - `docs/reference/index.md` lists: Datasources, Datasets, Datasets inner workings, Transforms, Evaluators, Model types, API, and (if present) How to add a model (or keep that in guides).

4. **Guides:**  
   - Move `docs/how_to_add_a_new_model.md` → `docs/guides/how_to_add_a_new_model.md`.  
   - Move `docs/runai.md` → `docs/guides/runai.md`.  
   - Update `docs/guides/index.md` and any cross-links.

**Exit criterion:** All reference and guide pages live under `reference/` and `guides/`; transforms are in one consistent place; reference index is complete; links updated.

---

## Phase 6: Webpage polish and branding

**Objective:** Make the doc set ready for a static site and consistent branding.

1. **Logo and branding:**  
   - Ensure piCID logo is used in `docs/index.md` (and optionally in `getting-started/index.md` or an “About” subsection).  
   - Add a short “About TabPHM / piCID” line if desired (e.g. PHM and forecasting benchmark arena; optional tagline).

2. **Navigation:**  
   - Add a simple “Doc nav” or “Contents” block in `index.md` and in each section index (getting-started, concepts, guides, examples, reference) pointing to sibling pages and back to main index.

3. **Link check:**  
   - Walk key paths: index → getting-started, concepts, examples, guides, reference; fix broken relative links.

4. **Optional (later):**  
   - Add `mkdocs.yml` or similar at repo root with nav matching the new structure; or add a note in `docs/plans/` that “Phase 6b: MkDocs” is for future site generation.

**Exit criterion:** Logo present, nav blocks in place, critical links valid; optional MkDocs left for a later iteration.

---

## Execution options

- **Sequential (recommended for first pass):** Execute Phase 1 → 2 → 3 → 4 in order; then Phase 5 (moves and link fixes); then Phase 6.  
- **Parallel where safe:** Phase 1 (structure) can be done first; Phase 2 and Phase 3 can be written in parallel after Phase 1; Phase 4 depends on Phase 3. Phase 5 and 6 after 2–4.  
- **Handoff:** A subagent or follow-up session can take “Phase N” as a single task with this plan as context; each phase’s exit criterion is the definition of done.

---

## Summary

| Phase | Key deliverables |
|-------|------------------|
| 1 | Folders, index with logo + sections, stub indexes, setup/viewer moved |
| 2 | Story page “From data to prediction”, concepts in order |
| 3 | Config/overrides guide, resolvers link |
| 4 | Unibo RUL example with commands and config snippets |
| 5 | Reference and guides moved, links fixed, transforms location decided |
| 6 | Logo/branding, nav, link check, optional MkDocs note |

This plan is intended to be updated as phases complete (e.g. checkboxes or “Done” notes in this file or in `docs/plans/`).

**Execution completed (2025-02-19):** Phases 1–6 implemented. New structure: getting-started/, concepts/, guides/, examples/, reference/. Logo at docs/assets/logo.png.
