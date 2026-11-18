"""Google-powered discovery: query builder, Custom Search JSON API client, Lens deep links.

Only Google's official index is used. No Meta/Facebook APIs, no scrapers, no login
bypassing: social and marketplace pages are reached purely through `site:` operators,
so only what Google has publicly indexed can ever appear.

RATE LIMITING & QUOTA (Custom Search JSON API)
----------------------------------------------
* Free tier: 100 queries/day per project; paid: USD 5 per 1,000, max 10,000/day.
  The daily counter resets at midnight Pacific Time.
* Each call returns at most 10 results; `start` pages up to result 100.
* Per-minute limit: 100 queries/minute per user.

What this module does about it:
* One search = a handful of queries (default 6): sites are OR-ed into groups
  instead of spending one query per site.
* Every response is cached in SQLite for CSE_CACHE_TTL_HOURS (default 24 h);
  repeating a search costs zero quota.
* A local counter refuses to call Google once CSE_DAILY_BUDGET is reached, so a
  runaway loop can never burn a paid quota.
* Calls are spaced by CSE_MIN_INTERVAL_SECONDS and retried with exponential
  backoff on HTTP 429 / 5xx.
"""
from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, quote_plus

from . import config
from .analyzer import analyze_hit, normalize_plate, plate_variants
from .models import QueryCache, Result, Search, add_usage, usage, utcnow

SOCIAL_SITES = ["instagram.com", "tiktok.com", "x.com"]
MARKETPLACE_SITES = ["facebook.com/marketplace", "facebook.com/groups"]
CLASSIFIED_SITES = ["clasipar.paraguay.com", "hendyla.com"]
CONTEXT_KEYWORDS = ['"chapa"', '"venta"', '"transferir"', '"contrato privado"', '"auto"']


class SearchError(Exception):
    pass


class QuotaExceeded(SearchError):
    pass


# --------------------------------------------------------------------------- #
# Query builder
# --------------------------------------------------------------------------- #
def _sites(sites: list[str]) -> str:
    return "(" + " OR ".join(f"site:{s}" for s in sites) + ")"


def _or(terms: list[str]) -> str:
    return terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"


def google_url(query: str) -> str:
    """Open the same query on google.com (Link mode: costs no API quota)."""
    return "https://www.google.com/search?q=" + quote_plus(query)


def lens_url(image_url: str) -> str:
    return "https://lens.google.com/uploadbyurl?url=" + quote(image_url, safe="")


def reverse_image_url(image_url: str) -> str:
    return "https://www.google.com/searchbyimage?client=app&image_url=" + quote(image_url, safe="")


def build_queries(params: dict, max_queries: int = 6) -> list[dict]:
    """Generate the search-operator permutations for a plate and/or vehicle description.

    Most specific queries come first so a small `max_queries` budget is spent well.
    """
    variants = plate_variants(params.get("plate", ""))
    plate_term = _or([f'"{v}"' for v in variants]) if variants else ""

    make, model = (params.get("make") or "").strip(), (params.get("model") or "").strip()
    year, color = (params.get("year") or "").strip(), (params.get("color") or "").strip()
    location = (params.get("location") or "").strip()
    vehicle = " ".join(x for x in (make, model) if x)
    vehicle_term = f'"{vehicle}"' if vehicle else ""
    extras = " ".join(x for x in (year, color, f'"{location}"' if location else "") if x)
    keywords = _or(CONTEXT_KEYWORDS)

    out: list[dict] = []

    def add(label: str, platform: str, *parts: str) -> None:
        q = " ".join(p for p in parts if p).strip()
        if q and all(q != o["query"] for o in out):
            out.append({"label": label, "platform": platform, "query": q, "google_url": google_url(q)})

    if plate_term:
        add("Plate · whole web", "Web", plate_term)
        add("Plate · Marketplace & groups", "Marketplace", plate_term, _sites(MARKETPLACE_SITES))
        add("Plate · classifieds", "Classifieds", plate_term, _sites(CLASSIFIED_SITES))
        add("Plate · social posts", "Social Post", plate_term, _sites(SOCIAL_SITES))
        add("Plate · sale / transfer keywords", "Web", plate_term, keywords)
        if vehicle_term or location:
            add("Plate · with vehicle details", "Web", plate_term, vehicle_term, f'"{location}"' if location else "")
    if vehicle_term:
        add("Vehicle · classifieds", "Classifieds", vehicle_term, extras, _sites(CLASSIFIED_SITES))
        add("Vehicle · Marketplace & groups", "Marketplace", vehicle_term, extras, _sites(MARKETPLACE_SITES))
        add("Vehicle · sale / transfer keywords", "Web", vehicle_term, extras, keywords)
        add("Vehicle · social posts", "Social Post", vehicle_term, extras, _sites(SOCIAL_SITES))
    elif not plate_term and location:
        add("Location · vehicle sales", "Web", f'"{location}"', keywords)
    return out[: max(1, max_queries)]


# --------------------------------------------------------------------------- #
# Custom Search JSON API client
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_last_call = 0.0


def is_configured() -> bool:
    return bool(config.env("GOOGLE_API_KEY") and config.env("GOOGLE_CSE_ID"))


def quota_day() -> str:
    """Google resets daily quotas at midnight Pacific Time."""
    return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def quota_status(db) -> dict:
    used = usage(db, "customsearch", quota_day())
    return {"used": used, "budget": config.cse_daily_budget(), "remaining": max(0, config.cse_daily_budget() - used)}


def _cse_call(query: str, start: int) -> dict:
    """One paced, retried call to customsearch v1."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    global _last_call
    service = build("customsearch", "v1", developerKey=config.env("GOOGLE_API_KEY"), cache_discovery=False)
    request = service.cse().list(q=query, cx=config.env("GOOGLE_CSE_ID"), num=10, start=start,
                                 gl=config.env("CSE_COUNTRY", "py"))
    for attempt in range(4):
        with _lock:  # pacing: never closer together than the configured interval
            wait = config.cse_min_interval() - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.monotonic()
        try:
            return request.execute(num_retries=0)
        except HttpError as exc:
            status = exc.resp.status
            reason = str(exc.reason or exc)
            if status == 429 and "per day" in reason.lower() or "dailyLimit" in reason:
                raise QuotaExceeded("Google daily Custom Search quota is used up. It resets at midnight Pacific Time.") from exc
            if status in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt * 1.5)  # 1.5 s, 3 s, 6 s
                continue
            if status == 429:
                raise QuotaExceeded("Google is rate-limiting Custom Search requests. Try again in a minute.") from exc
            if status in (400, 403):
                raise SearchError(f"Google rejected the request ({status}): {reason}. Check the API key and "
                                  "Search engine ID in Settings, and that the Custom Search API is enabled.") from exc
            raise SearchError(f"Custom Search failed ({status}): {reason}") from exc
    raise SearchError("Custom Search failed after retries.")


def cse_search(db, query: str, start: int = 1) -> tuple[dict, bool]:
    """Cached Custom Search call. Returns (response, served_from_cache)."""
    key = hashlib.sha1(f"{config.env('GOOGLE_CSE_ID')}|{start}|{query}".encode("utf-8")).hexdigest()
    cached = db.get(QueryCache, key)
    if cached and utcnow() - cached.fetched_at < timedelta(hours=config.cache_ttl_hours()):
        return cached.response, True

    if usage(db, "customsearch", quota_day()) >= config.cse_daily_budget():
        raise QuotaExceeded(f"Local daily budget of {config.cse_daily_budget()} Custom Search queries reached "
                            "(CSE_DAILY_BUDGET). Cached searches and Link mode still work.")
    response = _cse_call(query, start)
    add_usage(db, "customsearch", quota_day(), 1)
    slim = {"items": response.get("items", []), "searchInformation": response.get("searchInformation", {})}
    if cached:
        cached.response, cached.fetched_at = slim, utcnow()
    else:
        db.add(QueryCache(key=key, query=query, response=slim))
    db.commit()
    return slim, False


def _item_images(item: dict) -> tuple[str, str]:
    pagemap = item.get("pagemap") or {}
    thumb = ((pagemap.get("cse_thumbnail") or [{}])[0]).get("src", "")
    image = ((pagemap.get("cse_image") or [{}])[0]).get("src", "")
    return thumb or image, image or thumb


def store_hit(db, search: Search, params: dict, hit: dict, label: str, seen: dict[str, Result]) -> Result:
    """Insert a hit, or merge it into the row for the same URL (deduplication)."""
    fields = analyze_hit(params, hit.get("title", ""), hit.get("snippet", ""), hit["url"], hit.get("pagemap"))
    existing = seen.get(fields["url_hash"])
    if existing:
        if label not in existing.query_labels:
            existing.query_labels = existing.query_labels + [label]
        return existing
    row = Result(search_id=search.id, display_link=hit.get("display_link", ""), thumbnail=hit.get("thumbnail", ""),
                 image_url=hit.get("image_url", ""), query_labels=[label], **fields)
    db.add(row)
    seen[fields["url_hash"]] = row
    return row


def run_search(db, params: dict, max_queries: int = 6, pages: int = 1) -> tuple[Search, list[str], dict]:
    """Run every generated query through Custom Search and store analyzed, de-duplicated results."""
    if not is_configured():
        raise SearchError("Standard mode needs a Google API key and Search engine ID (Settings tab). "
                          "Without them, use Link mode.")
    queries = build_queries(params, max_queries)
    if not queries:
        raise SearchError("Enter a plate, a make/model, or a location to search.")

    search = Search(plate=normalize_plate(params.get("plate", "")), mode="api", queries=queries,
                    **{k: (params.get(k) or "").strip() for k in ("make", "model", "year", "color", "location")})
    db.add(search)
    db.commit()

    seen: dict[str, Result] = {}
    errors: list[str] = []
    stats = {"api_calls": 0, "cache_hits": 0}
    for q in queries:
        for page in range(max(1, min(pages, 10))):
            try:
                response, cached = cse_search(db, q["query"], start=1 + page * 10)
            except QuotaExceeded as exc:
                errors.append(str(exc))
                db.commit()
                return search, errors, stats
            except SearchError as exc:
                errors.append(f"{q['label']}: {exc}")
                break
            stats["cache_hits" if cached else "api_calls"] += 1
            items = response.get("items", [])
            for item in items:
                thumb, image = _item_images(item)
                store_hit(db, search, params, {
                    "url": item.get("link", ""), "title": item.get("title", ""), "snippet": item.get("snippet", ""),
                    "display_link": item.get("displayLink", ""), "pagemap": item.get("pagemap"),
                    "thumbnail": thumb, "image_url": image,
                }, q["label"], seen)
            if len(items) < 10:
                break
    db.commit()
    return search, errors, stats
