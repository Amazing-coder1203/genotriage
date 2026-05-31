# GenoTriage 🧬

**AI-powered clinical genomics variant analysis dashboard**

GenoTriage is a B.Tech final year project that provides real-time pathogenicity predictions for human genomic variants using a trained machine learning pipeline.

Colab Notebook: https://colab.research.google.com/drive/1goIMILCBP8PhumT4YJFbhu7sAgPvLpP_?usp=sharing

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
| ML Model | Scikit-learn pipeline with stacked ensembling of Random Forest + LightGBM + XGBoost (trained on hg38 variants) |
| Data Sources | gnomAD, ClinVar, PharmGKB, MyVariant.info |

<img width="1450" height="605" alt="image" src="https://github.com/user-attachments/assets/9061ef07-f80f-4847-bd36-41f7fc96e30a" />
<img width="719" height="361" alt="image" src="https://github.com/user-attachments/assets/d2b27976-985b-4e27-b5ce-90d74049b307" />
<img width="717" height="612" alt="image" src="https://github.com/user-attachments/assets/9000720f-7bc3-451d-8795-2af8e0a886e4" />
<img width="764" height="768" alt="image" src="https://github.com/user-attachments/assets/8f6685d1-5a99-45be-bf37-ddc6c8171bfd" />


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
