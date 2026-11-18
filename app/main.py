"""FastAPI application: API + single-page UI.

Modes (chosen automatically from what is available -- nothing to configure up front):
  link      no keys: builds the Google queries, you open them and paste back what you find
  standard  GOOGLE_API_KEY + GOOGLE_CSE_ID set: runs the queries through Custom Search
  advanced  signed in with Google: adds Cloud Vision OCR and Sheets/Drive export
"""
from __future__ import annotations

import shutil
import uuid
from urllib.parse import quote
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import __version__, config, google_auth, google_search, sheets_export, vision_ocr
from .analyzer import (PLATFORMS, build_analytics, classify_plate, format_plate, normalize_plate, parse_pasted)
from .models import ImageScan, Result, Search, get_db, init_db, usage, utcnow

app = FastAPI(title="Vehicle OSINT Aggregator", version=__version__)
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))
init_db()
app.mount("/crops", StaticFiles(directory=str(config.CROPS_DIR)), name="crops")


class SearchParams(BaseModel):
    plate: str = ""
    make: str = ""
    model: str = ""
    year: str = ""
    color: str = ""
    location: str = ""
    max_queries: int = 6
    pages: int = 1

    def as_dict(self) -> dict:
        return {k: getattr(self, k).strip() for k in ("plate", "make", "model", "year", "color", "location")}


class ImportBody(SearchParams):
    text: str
    search_id: int | None = None


class SettingsBody(BaseModel):
    google_api_key: str | None = None
    google_cse_id: str | None = None


class FolderBody(BaseModel):
    path: str
    plate: str = ""


class ExportBody(BaseModel):
    search_id: int


def _is_local(request: Request) -> bool:
    return (request.client.host if request.client else "") in ("127.0.0.1", "::1", "localhost")


def _search_payload(search: Search) -> dict:
    results = [_decorate(r.to_dict()) for r in sorted(search.results, key=lambda r: -(r.confidence or 0))]
    return {"search": search.to_dict(), "results": results, "analytics": build_analytics(results)}


def _decorate(r: dict) -> dict:
    """Add Google Lens / reverse-image deep links for the listing thumbnail."""
    image = r.get("image_url") or r.get("thumbnail")
    r["lens_url"] = google_search.lens_url(image) if image else ""
    r["reverse_image_url"] = google_search.reverse_image_url(image) if image else ""
    return r


def _get_search(db: Session, search_id: int) -> Search:
    search = db.get(Search, search_id)
    if not search:
        raise HTTPException(404, "Search not found.")
    return search


# --------------------------------------------------------------------------- #
# Pages, status, settings
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"version": __version__})


@app.get("/api/status")
def status(db: Session = Depends(get_db)):
    email = google_auth.signed_in_email()
    cse = google_search.is_configured()
    return {
        "version": __version__,
        "mode": "advanced" if email else "standard" if cse else "link",
        "signed_in": bool(email), "email": email,
        "oauth_client": google_auth.client_configured(),
        "cse_configured": cse,
        "api_key_hint": ("…" + config.env("GOOGLE_API_KEY")[-4:]) if config.env("GOOGLE_API_KEY") else "",
        "cse_id": config.env("GOOGLE_CSE_ID"),
        "easyocr": vision_ocr.easyocr_available(),
        "ocr_engine": "google-vision" if email else "easyocr" if vision_ocr.easyocr_available() else "none",
        "quota": {
            "customsearch": google_search.quota_status(db),
            "vision": {"used": usage(db, "vision", vision_ocr.vision_month()), "budget": config.vision_monthly_budget()},
        },
        "platforms": PLATFORMS,
    }


@app.post("/api/settings")
def save_settings(body: SettingsBody, request: Request):
    if not _is_local(request):
        raise HTTPException(403, "Settings can only be changed from this computer.")
    updates = {}
    if body.google_api_key is not None:
        updates["GOOGLE_API_KEY"] = body.google_api_key.strip()
    if body.google_cse_id is not None:
        updates["GOOGLE_CSE_ID"] = body.google_cse_id.strip()
    config.save_env(updates)
    return {"ok": True}


@app.post("/api/settings/client-secret")
async def upload_client_secret(request: Request, file: UploadFile = File(...)):
    if not _is_local(request):
        raise HTTPException(403, "Settings can only be changed from this computer.")
    try:
        kind = google_auth.save_client_secret(await file.read())
    except google_auth.AuthError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "client_type": kind}


# --------------------------------------------------------------------------- #
# Google sign-in
# --------------------------------------------------------------------------- #
def _redirect_uri(request: Request) -> str:
    return config.env("OAUTH_REDIRECT_URI") or str(request.base_url).rstrip("/") + "/auth/callback"


@app.get("/auth/login")
def auth_login(request: Request):
    try:
        return RedirectResponse(google_auth.start_login(_redirect_uri(request)))
    except google_auth.AuthError as exc:
        return RedirectResponse("/?auth_error=" + quote(str(exc)))


@app.get("/auth/callback")
def auth_callback(request: Request, state: str = "", error: str = ""):
    if error:
        return RedirectResponse("/?auth_error=" + quote(f"Google sign-in was cancelled ({error})."))
    try:
        google_auth.finish_login(state, str(request.url))
    except google_auth.AuthError as exc:
        return RedirectResponse("/?auth_error=" + quote(str(exc)))
    return RedirectResponse("/?signed_in=1")


@app.post("/auth/logout")
def auth_logout():
    google_auth.sign_out()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
@app.post("/api/queries/preview")
def preview_queries(body: SearchParams):
    plate = normalize_plate(body.plate)
    return {
        "plate": {"normalized": plate, "display": format_plate(plate), "format": classify_plate(plate)},
        "queries": google_search.build_queries(body.as_dict(), body.max_queries),
    }


@app.post("/api/search")
def run_search(body: SearchParams, db: Session = Depends(get_db)):
    try:
        search, errors, stats = google_search.run_search(db, body.as_dict(), body.max_queries, body.pages)
    except google_search.SearchError as exc:
        raise HTTPException(400, str(exc))
    db.refresh(search)
    return {**_search_payload(search), "errors": errors, "stats": stats}


@app.post("/api/import")
def import_text(body: ImportBody, db: Session = Depends(get_db)):
    """Link mode: analyze text/URLs the user copied from public pages and store them as results."""
    hits = parse_pasted(body.text)
    if not hits:
        raise HTTPException(400, "Nothing to import. Paste the text of a listing or search result, with its link.")
    params = body.as_dict()
    if body.search_id:
        search = _get_search(db, body.search_id)
        params = {k: getattr(search, k) for k in ("plate", "make", "model", "year", "color", "location")}
    else:
        search = Search(plate=normalize_plate(params["plate"]), mode="link",
                        queries=google_search.build_queries(params, body.max_queries),
                        **{k: params[k] for k in ("make", "model", "year", "color", "location")})
        db.add(search)
        db.commit()
    from .analyzer import url_hash
    seen = {r.url_hash: r for r in search.results}
    for h in hits:
        if url_hash(h["url"]) not in seen:
            google_search.store_hit(db, search, params, h, "Pasted", seen)
    db.commit()
    db.refresh(search)
    return {**_search_payload(search), "errors": [], "stats": {"imported": len(hits)}}


@app.get("/api/searches")
def list_searches(db: Session = Depends(get_db)):
    return [s.to_dict() for s in db.query(Search).order_by(Search.id.desc()).limit(100).all()]


@app.get("/api/searches/{search_id}")
def get_search(search_id: int, db: Session = Depends(get_db)):
    return _search_payload(_get_search(db, search_id))


@app.delete("/api/searches/{search_id}")
def delete_search(search_id: int, db: Session = Depends(get_db)):
    db.delete(_get_search(db, search_id))
    db.commit()
    return {"ok": True}


@app.get("/api/results")
def list_results(search_id: int | None = None, platform: str = "", match: str = "", min_confidence: float = 0.0,
                 q: str = "", db: Session = Depends(get_db)):
    """Server-side filtering (the UI filters client-side; this is for scripts)."""
    query = db.query(Result)
    if search_id:
        query = query.filter(Result.search_id == search_id)
    if platform:
        query = query.filter(Result.platform == platform)
    if match:
        query = query.filter(Result.match_type == match)
    query = query.filter(Result.confidence >= min_confidence)
    rows = [_decorate(r.to_dict()) for r in query.order_by(Result.confidence.desc()).limit(1000).all()]
    if q:
        needle = q.lower()
        rows = [r for r in rows if needle in (r["title"] + " " + r["snippet"] + " " + " ".join(r["locations"])).lower()]
    return rows


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
@app.post("/api/images/upload")
async def upload_images(files: list[UploadFile] = File(...), plate: str = Form(""), db: Session = Depends(get_db)):
    scans = []
    for f in files[:50]:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in config.IMAGE_EXTENSIONS:
            continue
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        scans.append(vision_ocr.analyze_image(db, dest, plate, original_name=f.filename or dest.name).to_dict())
    if not scans:
        raise HTTPException(400, "No supported images (jpg, png, webp, bmp).")
    return scans


@app.post("/api/images/scan-folder")
def scan_folder(body: FolderBody, request: Request, db: Session = Depends(get_db)):
    if not _is_local(request):
        raise HTTPException(403, "Folder scanning is only available from this computer.")
    try:
        files = vision_ocr.list_folder_images(body.path)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    if not files:
        raise HTTPException(400, "No images found in that folder.")
    return [vision_ocr.analyze_image(db, p, body.plate).to_dict() for p in files]


@app.get("/api/images")
def list_images(db: Session = Depends(get_db)):
    return [s.to_dict() for s in db.query(ImageScan).order_by(ImageScan.id.desc()).limit(200).all()]


@app.delete("/api/images")
def clear_images(db: Session = Depends(get_db)):
    db.query(ImageScan).delete()
    db.commit()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@app.post("/api/export/sheets")
def export_sheets(body: ExportBody, db: Session = Depends(get_db)):
    search = _get_search(db, body.search_id)
    try:
        return {"url": sheets_export.export_to_sheets(db, search)}
    except google_auth.AuthError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Google Sheets export failed: {exc}")


@app.get("/api/export/csv")
def export_csv(search_id: int, db: Session = Depends(get_db)):
    search = _get_search(db, search_id)
    body = sheets_export.results_csv([r.to_dict() for r in search.results])
    name = f"vehicle-osint-{search.plate or search.id}.csv"
    return Response("﻿" + body, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# --------------------------------------------------------------------------- #
# Demo data (clearly fictional) so every screen can be explored with no keys
# --------------------------------------------------------------------------- #
DEMO_HITS = [
    ("https://example.com/marketplace/item/1001", "Toyota Hilux 2015 4x4 - Villarrica",
     "Vendo Toyota Hilux 2015 color blanco, chapa ABCD 123, papeles al día, listo para transferir. "
     "Gs. 145.000.000. Llamar al 0981 555 010. Villarrica, Guairá. 12 mar 2025"),
    ("https://example.com/clasificados/hilux-2015-blanco", "Hilux 2015 blanca impecable | Clasificados (demo)",
     "Toyota Hilux 2015 blanco, único dueño. Chapa ABCD-123. USD 19.500 negociable. Contacto 0981-555-010 "
     "@autos_demo_py. Coronel Oviedo, Caaguazú. 2 abr 2025"),
    ("https://example.com/groups/compraventa/posts/88", "Compra venta de vehículos Guairá (demo)",
     "Se vende camioneta Hilux blanca, chapa ABCD123, con contrato privado. 140 millones. "
     "Escribir al 0971 555 020. Villarrica. 20 abr 2025"),
    ("https://example.com/noticias/vehiculo-recuperado", "Recuperan camioneta denunciada como robada (demo)",
     "Agentes verificaron una Toyota Hilux con chapa ABCO 123 en Asunción; el número de chasis no coincidía. "
     "15 may 2025"),
    ("https://example.com/social/p/xyz", "@autos_demo_py: Nueva oferta de la semana (demo)",
     "Hilux 2015 blanca full equipo, financiación propia. Consultas 0981 555 010. Asunción, Central. 28 may 2025"),
    ("https://example.com/clasificados/corolla-2012", "Toyota Corolla 2012 gris (demo)",
     "Corolla 2012 gris, chapa WXYZ 456, Gs. 62.000.000. Tel 0961 555 030. Encarnación, Itapúa. 3 jun 2025"),
]


@app.post("/api/demo")
def load_demo(db: Session = Depends(get_db)):
    params = {"plate": "ABCD 123", "make": "Toyota", "model": "Hilux", "year": "2015", "color": "blanco", "location": "Villarrica"}
    search = Search(plate="ABCD123", mode="demo", queries=google_search.build_queries(params),
                    **{k: params[k] for k in ("make", "model", "year", "color", "location")})
    db.add(search)
    db.commit()
    seen: dict = {}
    for i, (url, title, snippet) in enumerate(DEMO_HITS):
        row = google_search.store_hit(db, search, params, {"url": url, "title": title, "snippet": snippet,
                                                           "display_link": "example.com"}, "Demo data", seen)
        row.first_seen = utcnow() - timedelta(days=len(DEMO_HITS) - i)
    # The demo URLs are placeholders, so label platforms by what each entry represents.
    for row, platform in zip(seen.values(), ["Marketplace", "Classifieds", "Social Post", "News", "Social Post", "Classifieds"]):
        row.platform = platform
    db.commit()
    db.refresh(search)
    return {**_search_payload(search), "errors": [], "stats": {"demo": True}}
