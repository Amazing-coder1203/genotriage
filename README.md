# GenoTriage 🧬

**AI-powered clinical genomics variant analysis dashboard**

GenoTriage is a B.Tech final year project that provides real-time pathogenicity predictions for human genomic variants using a trained machine learning pipeline.

## Features

- 🔬 **Single Variant Analysis** — Input a variant (chr, pos, ref, alt) and get instant AI-driven pathogenicity predictions with confidence scores
- 📊 **Batch CSV Processing** — Upload a CSV of variants for bulk analysis with a live progress tracker
- 📈 **Rich Visualizations** — Radar charts, gauge charts, population frequency (gnomAD), in-silico damage scores
- 📋 **Downloadable Reports** — Export per-variant clinical PDF reports
- 🌙 **Dark / Light Mode** — Clean clinical dashboard UI with full dark mode support
- 🧠 **CDSS Panel** — Clinical Decision Support System recommendations per variant

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, Tailwind CSS, Chart.js |
| Backend | Python, FastAPI |
| ML Model | Scikit-learn pipeline (trained on hg38 variants) |
| Data Sources | gnomAD, ClinVar, PharmGKB, MyVariant.info |

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run locally
```bash
uvicorn main:app --reload
```

Then open `index.html` in your browser (or serve it via a static file server).

## Project Structure

```
├── index.html          # Frontend dashboard (single-page app)
├── main.py             # FastAPI backend with ML inference
├── requirements.txt    # Python dependencies
└── GenePath/           # Core ML pipeline module
```

## Genome Build

All variants are analyzed against **GRCh38 (hg38)**.

---

*B.Tech Project — Bioinformatics & Genomics AI*
