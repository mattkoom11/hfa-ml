# hfa-ml

Machine learning on UVA's Historic Clothing Collection: the learned related-item
recommender and semantic search planned in the vault page
`wiki/concepts/archive-recommender-ml-plan.md`. This repo covers Phase 0 (data
foundation) so far.

**Private.** The source spreadsheet holds donor names and emails, and the
collection data and photos belong to UVA. Everything under `data/` is
git-ignored. Only `collection_full.parquet` keeps donor columns; it exists for
the CollectiveAccess import, and nothing in the ML work reads it.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Put `Historic_Collection_Updated.xlsx` in `data/raw/`. The photo share (the NAS,
now the primary copy of the photos) lives at `/mnt/nas-garment-capture`; run
`sudo bash scripts/setup_nas_automount.sh` once so it mounts itself on first access.

## Refreshing after new garments or photos

```bash
.venv/bin/python -m hfa.refresh --dry-run   # what would run
.venv/bin/python -m hfa.refresh --deploy    # rebuild, gate, deploy to the VPS
```

Runs every step below in order, skipping steps whose outputs are newer than their
inputs. Before deploying, the gate runs the tests, `hfa.ci_check`, and integrity checks on
the new index: no personal-data columns, embeddings line up, and no more than 5% fewer garments than
are live. Deploy keeps the previous index on the VPS and rolls back if `/health`
doesn't report the new one. Afterwards, commit the frontend's `data/related.json`.
Splits stay frozen: existing garments keep theirs, and new garments are split among themselves.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Clean the spreadsheet | `.venv/bin/python -m hfa.clean` | `data/processed/collection_clean.parquet`, `data/reports/cleaning_issues.csv` |
| Index the photos | `.venv/bin/python -m hfa.manifest` | `data/interim/photo_manifest.parquet`, `data/processed/photo_coverage.parquet` |
| Join + freeze splits | `.venv/bin/python -m hfa.splits` | `data/processed/garments.parquet`, `data/reports/photo_orphans.csv`, MLflow run in `phase0-data` |
| RAW → JPG for RAW-only garments | `.venv/bin/python -m hfa.convert` | `data/interim/jpg/<accession>/*.jpg`, `data/interim/converted_jpgs.parquet` |
| Tests | `.venv/bin/python -m pytest -q` | |

## Experiment tracking

Every run logs to a local SQLite store at `mlruns/mlflow.db` (git-ignored).
Browse with `scripts/mlflow_ui.sh`, then open http://127.0.0.1:5000. Data
builds log the source spreadsheet's SHA-256, the split seed, row counts and the
issue reports, so any model result can be traced to the exact data it saw.

Splits are by garment, seeded, and stratified by clothing/accessory and CA
type. Don't regenerate them after Phase 1 starts. If the collection grows, add
the new garments with a new seed instead of reshuffling the existing ones.

## Open items

- 5 dates stay unparsed (`WWI` ×3, `?` ×2). Whole-century dates ("20thc") are
  tagged `century_whole` and excluded from decade labels.
- `cleaning_issues.csv` and `photo_orphans.csv` list what the archive team has to resolve.
