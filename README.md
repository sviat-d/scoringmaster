# Lead Scorer

Web app that scores and enriches outbound leads from CSV files by crawling company websites.

## Modes

- **inxy_leads** (primary) — Scores leads for Inxy crypto payment processing. Detects industries with high crypto adoption even without explicit crypto mentions.
- **founders_pl** (secondary) — Identifies product IT founders, rejects service companies.

## Quick Start

```bash
pip install -r requirements.txt
python -m app.main
```

Open http://localhost:8000, upload a CSV with a "Company Domain" column.

## Optional: LLM Enrichment

Set `OPENAI_API_KEY` env var to enable GPT-based classification on top of rules engine.

## Deploy (Railway)

```bash
railway up
```

Set env vars: `PORT` (auto), optionally `OPENAI_API_KEY`.
