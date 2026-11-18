# Vehicle Listing & OSINT Aggregator (Google Ecosystem Edition)

Finds public mentions of a license plate, make/model or location using only Google's
official index and APIs. No Meta/Facebook APIs, no scrapers, no login bypassing.

## Two ways to use it

| | Browser version (`web/`) | Full app (`app/`) |
|---|---|---|
| Where | GitHub Pages: **https://cesgod.github.io/vehicle-osint/** | Your computer: `run.bat` / `run.sh` |
| Setup | None | None for Link mode; Google credentials for the rest |
| Builds Google queries, Lens links | Yes | Yes |
| Paste-back analysis, filters, entities, timeline | Yes (saved in that browser) | Yes (saved in SQLite) |
| Runs searches automatically (Custom Search API) | No | Yes, with an API key |
| Cloud Vision OCR on photos, folder scan | No | Yes, with Google sign-in |
| Export to Google Sheets | Copy-paste into a sheet | One click, with Google sign-in |

GitHub Pages only serves static files, so the Python app cannot run there. The browser
version is redeployed automatically by `.github/workflows/pages.yml` whenever `web/` changes
on `main`.

Both UIs are mobile friendly. To open the full app from your phone on the same Wi-Fi, start
it with `HOST=0.0.0.0` and visit `http://<your-PC-IP>:8000`. Only do this on a trusted
network and while signed out of Google: the app has no login of its own.

## Start (no configuration)

Double-click **`run.bat`** (Windows) or run `./run.sh` (macOS/Linux). The first run creates
`.venv`, installs dependencies, then opens http://localhost:8000.

Needs Python 3.10+ (3.11+ recommended; Google's libraries warn on 3.10).

## Modes — picked automatically

| Mode | Needs | What you get |
|---|---|---|
| **Link** | nothing | Generated Google queries with one-click links, paste-back analysis, filters, analytics, timeline, CSV, demo data |
| **Standard** | API key + Search engine ID | The app runs the queries itself through Custom Search |
| **Advanced** | Google sign-in | Cloud Vision OCR on images, export to Google Sheets + Drive |

Keys are entered in the **Settings** tab (saved to `.env`, applied immediately).

## One-time Google Cloud setup (only for Standard / Advanced)

1. https://console.cloud.google.com → create a project.
2. **APIs & Services → Library**: enable *Custom Search API*, *Cloud Vision API*,
   *Google Sheets API*, *Google Drive API*. Vision requires a billing account on the
   project (first 1,000 units/month per feature are free).
3. **Standard mode**: *Credentials → Create credentials → API key*. Then create a search
   engine at https://programmablesearchengine.google.com with "Search the entire web" on,
   and copy its **Search engine ID**. Paste both in Settings.
   > Check Google's current terms first: Google has been restricting the Custom Search
   > JSON API for new customers. If you cannot get a key, Link mode does the same searches.
4. **Advanced mode**: *OAuth consent screen* → External, add yourself as a test user.
   *Credentials → Create credentials → OAuth client ID → Desktop app* → download the JSON.
   Upload it in Settings (or save it here as `client_secret.json`), then click
   **Sign in with Google**. If Google reports `redirect_uri_mismatch`, create a
   *Web application* client instead with redirect URI `http://localhost:8000/auth/callback`.

Scopes requested: `cloud-vision` and `drive.file` (only files this app creates).

## Rate limits & quotas

| API | Free allowance | How the app protects it |
|---|---|---|
| Custom Search | 100 queries/day (resets midnight PT) | Sites OR-ed into ~6 queries per search; 24 h SQLite cache; local `CSE_DAILY_BUDGET` stop; 1 s spacing; backoff on 429/5xx |
| Cloud Vision | 1,000 units/month per feature | 2 units per image; images downsized to 1600 px; local `VISION_MONTHLY_BUDGET` stop |
| Sheets | 60 writes/minute | 3 calls per export, plus one Drive upload per plate crop (max 25) |

All limits are tunable in `.env` (see `.env.example`). Details are in the module
docstrings of `app/google_search.py`, `app/vision_ocr.py`, `app/sheets_export.py`.

## Layout

```
app/main.py           FastAPI routes
app/google_auth.py    OAuth sign-in, token storage, Sheets/Drive/Vision clients
app/google_search.py  Query permutations, Custom Search client, Lens links
app/vision_ocr.py     OpenCV preprocessing, Cloud Vision / EasyOCR, cross-referencing
app/analyzer.py       Plate formats, fuzzy matching, entities, timeline
app/sheets_export.py  Google Sheets report, CSV
app/models.py         SQLite (SQLAlchemy)
templates/index.html  Tailwind + Alpine.js UI
templates/neu.css     Neumorphic theme (light/dark, phone layout) included by index.html
web/index.html        Browser-only version, deployed to GitHub Pages
.github/workflows/    Pages deployment
```

Optional offline OCR: `.venv\Scripts\python -m pip install easyocr==1.7.2` (about 2 GB).

## Use responsibly

Results are only what Google has publicly indexed. Verify matches on the original page,
and report suspected stolen vehicles to the police rather than confronting sellers.
The app is single-user and binds to 127.0.0.1; do not expose it to the internet while signed in.
