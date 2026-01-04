# Quantization Disparity Research Site

This directory contains the GitHub Pages site for the research project.

## Local Development

```bash
cd site
bundle install
bundle exec jekyll serve
```

Then visit `http://localhost:4000/quant-disparity/`

## Pages

- `index.md` - Home page with overview
- `tracks.md` - Research tracks (A, B, C, D)
- `experiments.md` - Experiment ledger with results
- `literature.md` - Literature review and paper digests
- `methodology.md` - Research methodology
- `roadmap.md` - What's next

## Deployment

Push to `main` branch. GitHub Pages will build from `/site` directory.

Configure in repo settings:
- Source: Deploy from branch
- Branch: main
- Folder: /site
