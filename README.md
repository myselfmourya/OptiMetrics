# OptiMetrics

**Optimizer Race: SGD vs AdaGrad vs Adam** — a mini project for 24AM501 Deep Learning.

`index.html` is a self-contained dashboard: a pre-computed NumPy benchmark (SGD vs AdaGrad vs Adam on the scikit-learn digits dataset) plus a **Live Lab** that trains all three optimizers from scratch, live, in the browser, on a dataset and hyperparameters you choose.

No build step — it's a static HTML file, so it deploys as-is on Netlify, GitHub Pages, or any static host.

## Contents

- `index.html` — the dashboard / prototype
- `optimizer_race.py` — the from-scratch NumPy benchmark script that produced the results shown in the dashboard
- `python_benchmark_results/` — the script's output (plots, metrics, logs)

## Deploying on Netlify

1. New site from Git → pick this repo.
2. Build command: *(leave blank)*
3. Publish directory: `.` (repo root)
4. Deploy — `index.html` is served at the site root automatically.
