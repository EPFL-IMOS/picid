# TabPHM Documentation Website

## Local development

1. Install deps: `uv sync`
2. Regenerate config map: `uv run picid-generate-web` (or `uv run python scripts/experiment_generator/generate_config_map.py`)
3. Serve: `uv run mkdocs serve`
4. Open: http://127.0.0.1:8000/

## Deploy to GitHub Pages

### Option A: Manual

```bash
uv run mkdocs gh-deploy
```

### Option B: GitHub Actions

The workflow `.github/workflows/docs.yml` (when added) runs on push to `main` when docs or configs change:

```yaml
# .github/workflows/docs.yml
name: Docs
on:
  push:
    branches: [main]
    paths: ['docs/**', 'mkdocs.yml', 'configs/**']
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run picid-generate-web
      - run: uv run mkdocs gh-deploy --force
```

Configure GitHub repo Settings → Pages → Source: Deploy from branch `gh-pages`.

## Build steps (in order)

```bash
uv run picid-generate-web   # update config map from configs/ (same as picid-assemble)
uv run mkdocs build         # build static site to site/
```
