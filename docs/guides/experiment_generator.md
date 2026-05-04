# Experiment Generator

Build Hydra CLI commands from dropdowns.

[**Open the Experiment Generator →**](../experiment-generator/index.html)

The generator reads available configs and lets you select experiment, paths, and debug options. Copy the generated command and run it in your terminal.

**Rebuilding the generator:** The web UI uses `config_map.json`, which is generated from `configs/` by `uv run picid-generate-web`. Run this before `mkdocs build` (or in CI before deploying docs) so the dropdown options stay in sync with the config tree. Same discovery logic as the CLI `picid-assemble`.

The UI artifact lives at `docs/experiment-generator/index.html`, and the generated data file lives at `docs/experiment-generator/config_map.json`.

After running, see [Submit Run to Leaderboard](submit_run_to_leaderboard.md) to upload and register results.
