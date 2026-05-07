# PICID Documentation Website

## Local development

1. Install deps: `uv sync`
2. Serve: `uv run mkdocs serve`
3. Open: http://127.0.0.1:8000/

## Deploy to GitHub Pages

### Option A: Manual

```bash
uv run mkdocs gh-deploy
```

### Option B: GitHub Actions

Add `.github/workflows/docs.yml` to auto-deploy on push to `main`:

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
      - run: uv run mkdocs gh-deploy --force
```

Configure GitHub repo Settings → Pages → Source: Deploy from branch `gh-pages`.

## Build steps (in order)

```bash
uv run mkdocs build   # build static site to site/
```
