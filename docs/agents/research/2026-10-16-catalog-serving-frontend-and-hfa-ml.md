---
date: 2026-10-16T00:10:38-04:00
researcher: Claude (for mattkoom11)
git_commit: "hfa-ml 80445353b20989c2da6512d05670c73ec8d5658a; UVA_Collectiveaccess_Frontend fcfc1d8ca257824b2d30759419aa7d7d2f79635b"
branch: main (both repos, in sync with origin)
repository: hfa-ml + UVA_Collectiveaccess_Frontend
topic: "How the uvahcc.org frontend loads garments today, and what hfa-ml holds that a catalog could be served from"
tags: [research, codebase, garments, catalog, collectiveaccess, search, related-items, admin, hfa-ml]
status: complete
last_updated: 2026-10-16
last_updated_by: Claude
---

# Research: How the frontend loads garments, and what hfa-ml could serve

**Date**: 2026-10-16T00:10:38-04:00
**Researcher**: Claude (for mattkoom11)
**Git Commit**: hfa-ml `8044535`, frontend `fcfc1d8`
**Branch**: main (both)
**Repository**: `hfa-ml` (private) and `UVA_Collectiveaccess_Frontend` (public)

> Stored in the private `hfa-ml` repo because it describes hfa-ml's data, including the names of donor columns. It contains no data values. Frontend paths below are relative to `UVA_Collectiveaccess_Frontend/`; hfa-ml paths are relative to `hfa-ml/`.

## Research Question

How does the frontend load and display garments today, end to end? That covers where the 9 demo garments come from, the data shape it expects, how images and 3D models are resolved, how search and ML related items are wired in, what it assumes about CollectiveAccess (CA), and how the admin routes work. It also covers what hfa-ml holds that a catalog could be served from: file locations, formats, record counts and fields, especially donor fields.

## Summary

- **The frontend has exactly two garment sources, chosen in `getAllGarments()`** (`lib/garments.ts:161-170`):
  - The CA cache, used when `CA_BASE_URL`, `CA_USERNAME` and `CA_PASSWORD` are all set and hydration has filled the cache.
  - Otherwise, the static `data/garments.json`.
  - No CA instance is configured, so the site serves the static file: **9 records, all 3D-scan entries with `featuredOnRunway: true`, local `/models/*.glb` files and empty `images` arrays**. No static record has a photo URL.
- **Photos reach the frontend only from CA.** They arrive as absolute URLs from CA media representations. `public/` holds no garment photos.
- **The ML layers are already keyed by accession number:**
  - `data/related.json` covers 1,202 accessions.
  - The search service returns accession numbers.
  - Both resolve through `Garment.accessionNumber`, so they only link to garments the site has loaded. Today 7 of the 9 loaded garments have an accession number. None of the 9 is among the 1,202 in `related.json`.
- **hfa-ml has no CA-import code and no catalog export.** What it does hold:
  - A cleaned catalog of 2,644 garments (`data/processed/garments.parquet`, no personal columns). 1,202 of them have photos.
  - 5,821 local JPGs (1,600 px long edge, about 1.5 GB) under `data/interim/jpg/<accession>/`.
  - A search index (`data/exports/search_index/garments.csv`, 1,202 rows).
  - Attribute predictions in `data/predictions/`.
  - Donor data lives only in `data/interim/collection_full.parquet`, in the four columns listed in `PII_COLUMNS`.

## Detailed Findings

### 1. Frontend data layer

**Static file.** `data/garments.json` is an array of 9 flat objects:
- Test dresses 0–4, plus `dr-2023-17`, `dr-2023-323`, `ens-2023-14` and `dr-2026-118`.
- Keys present in all 9: `id, slug, label, name, era, type, work_type, colors, materials, images, model3d_url, featuredOnRunway, model3d_rotationY, collection, description`.
- `accessionNumber` is present in 7 of the 9.
- The `collection` value is `"Demo"`, and `colors`, `materials` and `images` are empty.
- The only code import is `lib/garments.ts:1`.

**Types** (`types/garment.ts`):
- `Garment` (37-85) requires `id, slug, label, images`. All other fields are optional, including `accessionNumber`, `models`, `featuredOnRunway`, `runwayOrder` and `storageLocation`.
- `Era` (line 2) has four buckets. `GarmentType` (5-24) has 19 members.
- Helpers: `getEraFromDecade` (88-111), `normalizeMaterials` (114-118) and `getGarmentTypeFromWorkType` (121-216).

**`lib/garments.ts`:**
- **Caches:** an in-memory `caGarmentsCache` (5-8) and a disk cache at `data/ca-garments-cache.json` (10-52). The disk cache is written atomically, is git-ignored, and **does not exist on disk** right now.
- **`hydrateGarmentsFromCA()`** (107-151):
  - Does nothing unless CA is configured.
  - Tries the disk cache first.
  - Otherwise runs `syncGarmentsFromCA(0, true)` (metadata only) with a 30 s timeout.
  - On failure it falls back to static data and waits 60 s before trying again.
  - Missing photos are filled in later by `fillMissingImagesInBackground` (80-100).
- **Search and filters:** `searchGarments` (228-270) is the keyword search. `filterGarments` (181-222) drives the collection filters.

**Callers:**

| Caller | How it reads garments |
|---|---|
| `app/page.tsx`, `app/collection/page.tsx`, `app/garments/[slug]/page.tsx` | Server components with `force-dynamic`. They start hydration without awaiting it, then call `getAllGarments()`. |
| `app/timeline/page.tsx:8` | The only page that awaits hydration. |
| `app/statistics`, `app/compare` | Call `getAllGarments()` without hydrating. They render dynamically because the root layout reads `headers()` for the CSP nonce (`app/layout.tsx:53-62`). |
| `app/api/garments/route.ts:4-8` | Awaits hydration, then returns every garment as JSON. |
| `GarmentSearchClient.tsx:8`, `ComparePageClient.tsx:29`, `ExhibitionsPageClient.tsx:5` | Client components that import `lib/garments` directly. In the browser the CA cache is null, so they always see the static file. |

`data/sampleGarments.ts` defines its own, separate `Garment` interface. Only `data/shows.ts` imports it, and nothing imports `data/shows.ts`.

### 2. Images and 3D models

**Photos:**
- Static garments have no photo fields. CA garments get them from `fetchObjectImages` (`lib/collectiveAccess.ts:285-311`), which reads `ca_objects.representations`:
  - `url` is `urls.original`.
  - `thumbnail_url` is `urls.preview170`, falling back to `thumbnail`, then `original`.
- `convertToGarment` sets `images`, `imageUrl` and `thumbnailUrl` (520-522). URLs are used exactly as CA returns them, with no base-URL prefix.

**Config and CSP:**
- `next.config.ts` sets `images.unoptimized: true` (36), with the comment that CA media may need a session cookie.
- `remotePatterns` comes only from the CA hostnames (9-28).
- In `proxy.ts`, `img-src` allows any `https:` or `http:` host (27).
- `connect-src` is `'self' blob:` (37), which allows no external hosts. That means a GLB file on another host would be blocked.

**How each component picks a photo:**
- `GarmentCard` and `GarmentImage` try `thumbnailUrl`, then `imageUrl`, then `images[0]`, and show an icon placeholder on error.
- `Lightbox`, the hall's wall panel, tries `imageUrl`, then `images[0]`, then `thumbnailUrl`.
  - It loads the photo as a WebGL texture once the visitor is within 22 units (`components/museum/Lightbox.tsx:11-68`).
  - The photo host must send CORS headers. If it doesn't, the panel stays blank cream.
- `ImageGallery` renders `garment.images` and returns nothing when the array is empty.

**3D models:**
- The static records point at local files: `public/models/` holds 9 GLBs, plus one unreferenced `testdress0-lowpoly.glb`.
- For CA records, `object_3d_documentation` entries become `models[]` (`parseModelEntries`, `lib/collectiveAccess.ts:64-78`), each with a role of `web_preview`, `full_detail` or `archival_master`. Only URLs starting with `http(s)://` or `/` are kept.
- `lib/museum.ts` resolves which file to load:
  - `getPreviewModelUrl` (40-42): the `web_preview` file, else `model3d_url`, else `modelUrl`.
  - `getDetailModelUrl` (45-48): the `full_detail` file, else the preview.
  - `isFlagship` (55-57): `featuredOnRunway` is true and a preview URL exists.
- `buildMuseumLayout` (99-124) sorts garments into era wings:
  - Flagships go on the runway.
  - Garments with a photo become wall lightboxes.
  - Everything else goes to `unplaced`.
- `FlagshipGarment` switches to the `full_detail` model while a garment is being inspected (`components/museum/FlagshipGarment.tsx:71, 95-96`).
- `Garment3DViewer` loads `model3d_url` on the detail page. Without one it renders a procedural placeholder.

### 3. Search and related items

**Search:**
- `/api/search` (`app/api/search/route.ts:5-17`) first calls `semanticSearch` (`lib/semanticSearch.ts`):
  - The URL comes from `SEARCH_SERVICE_URL` and the token from `SEARCH_SERVICE_TOKEN`. Both are environment-only, and no hostname appears in the repo.
  - It sends `POST {url}/search` with body `{q, k: 24}`, the query cut to 200 characters, a Bearer token and a 2 s timeout.
  - It expects `{results: string[]}` back: a list of accession numbers.
  - It maps each one through `accessionNumber`, drops any that aren't loaded, and returns `null` if none match or anything fails.
- In that `null` case the route falls back to `searchGarments` keyword search. The response says which engine answered (`"semantic"` or `"keyword"`).
- `/collection` doesn't use this route. It runs `advancedSearch` (`lib/advancedSearch.ts`) in the browser.

**Related items:**
- `data/related.json` (generated 2026-10-10 by `hfa.export_related`, `k: 8`) maps each of 1,202 accession numbers to up to 8 related accessions, best first.
- `getEnhancedRelatedGarments` (`lib/relatedGarments.ts:102-139`) fills the detail page's 4 slots in this order:
  1. The curator's `relatedIds`.
  2. `related.json` entries that are loaded (`lib/relatedPrecomputed.ts:29-50`).
  3. The weighted score (era 0.3, type 0.25, color 0.2, material 0.15, decade 0.1) for whatever slots remain.
- A garment with no `related.json` entry gets the weighted score. A related accession that isn't loaded is skipped.

### 4. CollectiveAccess assumptions

**Auth flow** (`lib/collectiveAccess.ts:107-180`):
1. Load the CA web login form and scrape its CSRF token.
2. POST the form to `DoLogin` to get the `collectiveaccess=` session cookie.
3. Call `service.php/json/auth/login` with the cookie and Basic Auth to get an `authToken`, which is cached for 23 h.

**Data requests:** calls go to `service.php/json/...` with a 20 s timeout and a 5-minute in-memory cache. The endpoints are:
- `/find/ca_objects`
- `/item/ca_objects/id/{id}`
- `/find/ca_sets` and `/item/ca_sets/id/{id}`, used when `CA_SET_CODE` is set
- `fetchAllObjects` pages through 100 records at a time, up to 50 pages, and fetches each object's detail with concurrency 10.

**Bundles read** (`DETAIL_BUNDLES`, 675-681): `preferred_labels`, `type_id`, `idno`, `date_range`, `condition`, `storage_location`, `gender`, `age_group`, `color_location`, `material_location`, `function`, `description`, `web_narrative`, `provenance`, `web_display_settings`, `object_3d_documentation`, and `representations`.

**How CA fields become `Garment` fields** (`convertToGarment`, 425-532):
- `idno` becomes `accessionNumber`.
- `object_id` becomes `id`.
- `generateSlug(idno)` becomes `slug`.
- `web_narrative.{web_story, pull_quote, style_notes}` become `story`, `tagline` and `aesthetic_description`.
- `web_display_settings.featured_on_runway` and `homepage_order` become `featuredOnRunway` and `runwayOrder`.
- `collection` is always set to `'UVA Historic Clothing Collection'`.
- If a record has no type, `IDNO_PREFIX_TYPE` (382-423) supplies one from the idno prefix.

**Public-display filter:**
- `isPublic()` (615-618) reads `web_display_settings.public_display` and is fail-closed: an unset value means not public.
- The bundle name has not been checked against a live install.
- Setting `CA_SKIP_PUBLIC_DISPLAY_FILTER=true` bypasses the filter.

**Profile docs:**
- `docs/ca-profile-changes.md` (2026-09-30, proposed, not applied) lists profile edits and open decisions:
  - `access` versus `public_display`
  - the flagship rule
  - `idno` versus `accession_number`
  - `web_slug`
  - a read-only CA account for the site
- `docs/phase-5-plan.md` (2026-09-30, not started) plans deep links into the hall and a runway-map timeline.

### 5. Admin routes

**The `/admin` page:**
- `app/admin/page.tsx` wraps an inline `AdminDashboard` in `AdminAuthGate`.
- The dashboard has stat tiles, a "Sync CA data" button that POSTs to `/api/admin/sync`, CSV/JSON export, and analytics from localStorage.
- The page shell itself is public. Only the `/api/admin/*` calls it makes are gated.

**The two admin API routes:**

| Route | Methods | What it does |
|---|---|---|
| `app/api/admin/auth/route.ts` | `GET` | Checks the session cookie. |
| | `POST` | Checks the password and issues the cookie. Returns 429 when rate-limited and 400 on bad input. |
| | `DELETE` | Signs out. |
| `app/api/admin/sync/route.ts` | `POST` | Requires a valid session. Runs `syncGarmentsFromCA(0)`, a full sync with images, then `setCAGarmentsCache`, which also writes the disk cache. |

**Session handling** (`lib/adminAuth.ts`):
- The session token is `"<timestamp>.<HMAC-SHA256>"`, keyed with the admin password and valid for 4 h.
- The cookie is `uva_admin_session`: `HttpOnly`, `SameSite=Strict`, `Path=/api/admin`, and `Secure` in production.
- Password checks use timing-safe compares.
- Rate limits are 10 attempts per IP and 50 globally per 15 minutes, held in memory.
- `ADMIN_PASSWORD` is resolved on each call and is required in production.

### 6. What hfa-ml holds

**The pipeline:**
```
hfa.clean → hfa.manifest → hfa.convert (raw/web) → hfa.splits → hfa.embed
         → hfa.attr_keyword → hfa.attr_image probe → hfa.export_related, hfa.search_index
```
It runs end to end through `hfa.refresh` (`hfa/refresh.py:54-77`). That gate runs pytest, `hfa.ci_check` and index-integrity checks. With `--deploy` it rsyncs the search index to the VPS and rolls back if `/health` doesn't report the new count.

**Tables** (row counts measured 2026-10-16; no values were read):

| File | Rows | Columns |
|---|---|---|
| `data/interim/collection_full.parquet` | 2,658 | 28. **Contains `donor_raw`, `donor_name`, `donation_date_text` and `email`.** Its docstring says it is for the CA import only (`hfa/clean.py:4`), and no ML step reads it. |
| `data/processed/collection_clean.parquet` | 2,658 | 24. The full table minus `PII_COLUMNS` (`hfa/clean.py:37, 165`). |
| `data/processed/garments.parquet` | 2,644 | 34. The clean table restricted to unique, non-null accessions, joined to photo coverage, plus `split`. |
| `data/processed/photo_coverage.parquet` | 1,251 | `accession, n_files, n_shots, n_web, n_raw, n_folders, raw_only` |
| `data/interim/photo_manifest.parquet` | 10,639 | One row per file on the NAS. Paths are relative to `/mnt/nas-garment-capture`. |
| `data/interim/cached_jpgs.parquet` / `converted_jpgs.parquet` | 4,591 / 1,230 | Manifest columns plus `status` and `jpg_rel` |
| `data/exports/search_index/garments.csv` | 1,202 | `accession, title, ca_type, date_raw, date_earliest, date_latest, description, material, color, technique` |

**What's in `garments.parquet`:**
- 1,835 clothing records and 809 accessories.
- 1,202 have photos (`has_photos`, and also `n_ml_jpg > 0`). 1,433 have a description.
- Its columns include `title, ca_type, gender, age_group, date_raw, date_earliest, date_latest, condition, description, maker_label, storage_location, location_raw, old_number, group, has_description, has_photos`.

**Photos:**
- 5,821 JPGs across 1,251 accessions, about 1.5 GB, at `data/interim/jpg/<accession>/<stem>.jpg`.
- They are 1,600 px on the long edge and sRGB, made by `hfa/convert.py`.
- `--web` copies at most 8 shots per garment.
- These files exist only on this machine. No module publishes them anywhere.

**Attributes:**
- `data/predictions/keyword.json` holds rule-based material, color and technique codes for described garments, shaped `{accession: {material: [codes], color: [...], technique: [...]}}`. It comes from `hfa/attr_keyword.py` with the vocabulary in `hfa/vocab.json`.
- `data/predictions/image_probe*.json` holds photo-probe codes for photographed garments (`hfa/attr_image.py`).
- `hfa/recsys.py:57-79` (`load_catalog`) merges the two and records `attr_source` per garment.

**Embeddings:** `data/processed/garment_embeddings.npz` holds 512-dimension FashionCLIP vectors per garment (`hfa/embed.py`).

**What leaves the repo:**

| Export | Where it goes |
|---|---|
| `data/exports/related.json` | Copied to the frontend's `data/related.json` (`refresh.py:33`, `export_related.py:73-75`). Committing it is a manual step. |
| `data/exports/search_index/` | Rsynced to the VPS and served by `hfa/serve.py` (see below). |

**The search service** (`hfa/serve.py`):
- A FastAPI app.
- `GET /health` returns `{ok, garments}` and needs no authentication.
- `POST /search` takes `{q, k ≤ 24}` with a Bearer token and returns accession numbers only. It is rate-limited to 30 requests per minute per IP.
- Ranking fuses BM25 and CLIP results.

**Personal-data guards:**
- `FORBIDDEN = {donor, donor_raw, donor_name, donation_date_text, email}` appears in `hfa/search_index.py:28` (an assert when the index is built) and in `hfa/refresh.py:38` (the gate's column check on `garments.csv`).
- Neither check covers `storage_location`, `location_raw` or `old_number`. Those columns are in `collection_clean.parquet`, and the search index does not include them.

**CollectiveAccess:**
- No code produces CA import files.
- `data/raw/intake_mapping_sheet.xlsx` exists, and no code references it.

## Code References

**Frontend:**
- `lib/garments.ts:161-170`: `getAllGarments()` chooses CA cache or static
- `lib/garments.ts:107-151`: hydration (disk cache, 30 s sync, 60 s cooldown)
- `lib/collectiveAccess.ts:425-532`: `convertToGarment`, the CA → Garment mapping
- `lib/collectiveAccess.ts:615-618`: `isPublic()`
- `lib/semanticSearch.ts:26-52`: the external search call, with accession mapping
- `lib/relatedGarments.ts:102-139`: related-item layering
- `lib/relatedPrecomputed.ts:20-50`: `related.json` lookup
- `lib/museum.ts:40-57, 99-124`: model URL resolution, flagship rule, wing layout
- `proxy.ts:27, 37`: CSP `img-src` and `connect-src`
- `lib/adminAuth.ts:16-152`: admin session and rate limiting

**hfa-ml:**
- `hfa/clean.py:25-37, 160-166`: column renames, `PII_COLUMNS`, outputs
- `hfa/splits.py:57-79`: builds `garments.parquet`, frozen splits
- `hfa/convert.py:22-83`: local JPG cache
- `hfa/search_index.py:28-45`: index build and `FORBIDDEN` assert
- `hfa/serve.py:35-141`: search API
- `hfa/export_related.py:28-75`: `related.json` export
- `hfa/refresh.py:33-38, 54-77, 120-188`: step graph, gate, deploy

## Architecture Documentation

- **Join key.** The accession number (CA `idno`) is the shared key across both repos: search results, `related.json` keys and values, photo folders on the NAS, and `garments.parquet`. The frontend's `id` and `slug` come from CA's `object_id` and `idno` when CA is the source, and are hand-written in the static file.
- **Two separate garment worlds in the frontend.** Server components see the CA cache whenever it is populated. Client components that import `lib/garments` always see the static file.
- **Personal data stays inside hfa-ml.** Donor columns exist in one interim file. The clean table, `garments.parquet`, the search index, MLflow logs and `related.json` don't carry them. Two separate `FORBIDDEN` checks guard the index.
- **Photos have no public host yet.** The frontend displays photos only from CA representation URLs. hfa-ml's JPG cache and the NAS are local.

## Related Research

None yet in `docs/agents/research/`. The vault's pages `historic-clothing-collection-workflow` and `archive-recommender-ml-plan` cover the same system from the planning side.

## Open Questions

- Which photo host a non-CA catalog would use: the Lightbox needs CORS, and the CSP `img-src` already allows any host.
- Whether `storage_location`, `location_raw`, `old_number` and `maker_label` are fit for a public catalog. None of them is in `PII_COLUMNS`.
- How the 9 static 3D records (some without accession numbers) would merge with a larger catalog keyed by accession.
- Whether client components that read the static file (`GarmentSearchClient`, `ComparePageClient`, `ExhibitionsPageClient`) would see a new data source.
- The live CA bundle names (`public_display` and others) are still unverified (`docs/ca-profile-changes.md` decisions 6–12).
