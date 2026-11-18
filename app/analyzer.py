"""Plate normalization, fuzzy matching, entity extraction and analytics.

Pure text logic -- no network, no Google calls -- so it works in every mode.

Plate formats handled (Paraguay):
  * Mercosur car        ABCD 123   (4 letters + 3 digits)
  * Mercosur motorcycle 123 ABCD   (3 digits + 4 letters)
  * Legacy              ABC 123    (3 letters + 3 digits)
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlparse

from rapidfuzz import fuzz

# --------------------------------------------------------------------------- #
# Plates
# --------------------------------------------------------------------------- #
SEP = r"[\s\-\.·]?"
B0, B1 = r"(?<![A-Z0-9])", r"(?![A-Z0-9])"
PLATE_PATTERNS = [
    ("mercosur", re.compile(rf"{B0}([A-Z]{{4}}){SEP}(\d{{3}}){B1}")),
    ("mercosur-moto", re.compile(rf"{B0}(\d{{3}}){SEP}([A-Z]{{4}}){B1}")),
    ("legacy", re.compile(rf"{B0}([A-Z]{{3}}){SEP}(\d{{3}}){B1}")),
]
LOOSE_PLATE = re.compile(rf"{B0}([A-Z0-9]{{3,4}}){SEP}([A-Z0-9]{{3,4}}){B1}")

# Words that look like plates in listings ("BMW 320", "FIAT 147", "AÑO 201x").
PLATE_STOPWORDS = {
    "BMW", "KIA", "GOL", "FIAT", "AUDI", "FORD", "JEEP", "SEAT", "MINI", "AUTO", "MOTO", "FONO",
    "CASA", "RUTA", "TEL", "CEL", "USD", "DEL", "CON", "POR", "LOS", "LAS", "UNA", "ANO", "KMS",
    "GSM", "HP", "CC", "VAN", "SUV", "RAM", "BYD", "JAC", "GMC", "MODE", "TIPO", "PARA", "TODO",
    "CADA", "SOLO", "DESDE", "HILUX", "CRV", "RAV", "AMG", "GLA", "GLC", "GLE", "CLA", "TDI",
    "NRO", "NUM", "LOTE", "CALL", "ZONA", "CASI", "ESQ", "AVDA", "CUOTA",
}

# OCR confusions. Letters that are misread as digits and vice versa.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6", "T": "7"}
TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G", "7": "T"}
PLATE_TEMPLATES = [("mercosur", "LLLLDDD"), ("mercosur-moto", "DDDLLLL"), ("legacy", "LLLDDD")]


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_plate(text: str) -> str:
    """'abcd-123' -> 'ABCD123'."""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(text or "").upper())


def classify_plate(norm: str) -> str | None:
    if re.fullmatch(r"[A-Z]{4}\d{3}", norm):
        return "mercosur"
    if re.fullmatch(r"\d{3}[A-Z]{4}", norm):
        return "mercosur-moto"
    if re.fullmatch(r"[A-Z]{3}\d{3}", norm):
        return "legacy"
    return None


def split_plate(norm: str) -> list[str]:
    """'ABCD123' -> ['ABCD', '123'] (runs of letters / digits)."""
    return re.findall(r"[A-Z]+|\d+", norm)


def format_plate(norm: str) -> str:
    return " ".join(split_plate(norm))


def plate_variants(plate: str) -> list[str]:
    """All the ways people type a plate: 'ABCD123', 'ABCD 123', 'ABCD-123'."""
    norm = normalize_plate(plate)
    parts = split_plate(norm)
    if len(parts) < 2:
        return [norm] if norm else []
    out = []
    for sep in (" ", "", "-"):
        v = sep.join(parts)
        if v not in out:
            out.append(v)
    return out


def coerce_plate(token: str) -> tuple[str, str, int] | None:
    """Fit an OCR token to a plate format, fixing 0/O, 1/I, 5/S, 8/B ... by position.

    Returns (plate, format, corrections) or None.
    """
    tok = normalize_plate(token)
    best = None
    for fmt, template in PLATE_TEMPLATES:
        if len(tok) != len(template):
            continue
        chars, fixes, ok = [], 0, True
        for ch, kind in zip(tok, template):
            if kind == "L":
                if ch.isalpha():
                    chars.append(ch)
                elif ch in TO_LETTER:
                    chars.append(TO_LETTER[ch]); fixes += 1
                else:
                    ok = False; break
            else:
                if ch.isdigit():
                    chars.append(ch)
                elif ch in TO_DIGIT:
                    chars.append(TO_DIGIT[ch]); fixes += 1
                else:
                    ok = False; break
        if ok and (best is None or fixes < best[2]):
            best = ("".join(chars), fmt, fixes)
    return best


def _ocr_canonical(norm: str) -> str:
    """Collapse confusable characters so 'ABCD1O3' and 'ABCD103' compare equal."""
    return "".join(TO_DIGIT.get(c, c) for c in norm)


def find_plates(text: str) -> list[str]:
    """Strict plate-shaped strings in free text, normalized and de-duplicated."""
    up = strip_accents(text or "").upper()
    found: list[str] = []
    for _fmt, rx in PLATE_PATTERNS:
        for m in rx.finditer(up):
            letters = m.group(1) if m.group(1).isalpha() else m.group(2)
            if letters in PLATE_STOPWORDS:
                continue
            norm = m.group(1) + m.group(2)
            if norm not in found:
                found.append(norm)
    return found


def match_plate(target: str, text: str) -> tuple[str, float, str]:
    """Compare a target plate against free text.

    Returns (match_type, score 0-100, matched_text) where match_type is
    'exact', 'fuzzy' or 'none'.
    """
    t = normalize_plate(target)
    if not t or not text:
        return "none", 0.0, ""
    up = strip_accents(text).upper()

    exact = re.compile(B0 + SEP.join(re.escape(p) for p in split_plate(t)) + B1)
    m = exact.search(up)
    if m:
        return "exact", 100.0, m.group(0)

    best_score, best_text = 0.0, ""
    t_canon = _ocr_canonical(t)
    for m in LOOSE_PLATE.finditer(up):
        cand = m.group(1) + m.group(2)
        if abs(len(cand) - len(t)) > 1 or cand.isalpha() or cand.isdigit():
            continue
        score = 92.0 if _ocr_canonical(cand) == t_canon else fuzz.ratio(cand, t)
        if score > best_score:
            best_score, best_text = score, m.group(0)
    if best_score >= 83:  # one wrong character in a 6-7 character plate
        return "fuzzy", best_score, best_text
    return "none", 0.0, ""


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #
PHONE_RX = re.compile(r"(?<!\d)(?:\+?595[\s.\-]?|0)9[6-9]\d[\s.\-]?\d{3}[\s.\-]?\d{3}(?!\d)")
PRICE_RX = re.compile(
    r"(?:(?:GS\.?|₲|G\$|USD|US\$|U\$S|\$)\s?\d[\d.,]*(?:\s?(?:millones|mill\b|M\b))?"
    r"|\d(?:[\d.,]*\d)?\s?(?:millones|guaran[ií]es|GS\b|₲|USD\b))",
    re.IGNORECASE,
)
HANDLE_RX = re.compile(r"(?<![\w@.])@([A-Za-z0-9_.]{2,29}[A-Za-z0-9_])")
YEAR_RX = re.compile(r"(?<!\d)(19[6-9]\d|20[0-3]\d)(?!\d)")

LOCATIONS = [
    "Asunción", "Central", "Villarrica", "Guairá", "Ciudad del Este", "Alto Paraná", "Encarnación",
    "Itapúa", "San Lorenzo", "Luque", "Capiatá", "Lambaré", "Fernando de la Mora", "Limpio",
    "Ñemby", "Mariano Roque Alonso", "Itauguá", "Villa Elisa", "Caaguazú", "Coronel Oviedo",
    "Pedro Juan Caballero", "Amambay", "Concepción", "Caacupé", "Cordillera", "Paraguarí",
    "Pilar", "Ñeembucú", "Misiones", "San Juan Bautista", "Caazapá", "San Pedro", "Canindeyú",
    "Salto del Guairá", "Presidente Hayes", "Villa Hayes", "Boquerón", "Filadelfia",
    "Alto Paraguay", "Hernandarias", "Presidente Franco", "Minga Guazú", "Itá", "Areguá",
    "Ypacaraí", "Santa Rita", "Hohenau", "Obligado", "Curuguaty",
]
MAKES = [
    "Toyota", "Nissan", "Hyundai", "Kia", "Chevrolet", "Volkswagen", "Ford", "Mitsubishi", "Honda",
    "Mercedes-Benz", "Mercedes", "BMW", "Audi", "Fiat", "Renault", "Peugeot", "Suzuki", "Mazda",
    "Isuzu", "Subaru", "Jeep", "Chery", "JAC", "BYD", "Great Wall", "Volvo", "Scania", "Daihatsu",
    "Citroën", "Lexus", "Land Rover", "Yamaha", "Kenton", "Star", "Leopard", "Taiga",
]
COLORS = [
    "blanco", "negro", "gris", "plata", "plateado", "rojo", "azul", "verde", "amarillo", "naranja",
    "marrón", "beige", "dorado", "bordó", "celeste", "champagne", "white", "black", "silver",
    "gray", "red", "blue", "green",
]


def _word_list_rx(words: list[str]) -> re.Pattern:
    alts = sorted({strip_accents(w).lower() for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in alts) + r")(?![a-z0-9])")


_LOC_RX, _MAKE_RX, _COLOR_RX = _word_list_rx(LOCATIONS), _word_list_rx(MAKES), _word_list_rx(COLORS)
_CANON = {strip_accents(w).lower(): w for w in LOCATIONS + MAKES + COLORS}

_URL_USER_RULES = [
    (re.compile(r"facebook\.com/groups/([^/?#]+)", re.I), "fb-group:{}", set()),
    (re.compile(r"facebook\.com/([^/?#]+)", re.I), "fb:{}",
     {"marketplace", "groups", "watch", "photo", "photos", "permalink.php", "profile.php", "story.php",
      "share", "pages", "people", "events", "reel", "login", "sharer", "hashtag", "public"}),
    (re.compile(r"instagram\.com/([^/?#]+)", re.I), "ig:@{}", {"p", "reel", "reels", "explore", "tv", "stories"}),
    (re.compile(r"tiktok\.com/@([^/?#]+)", re.I), "tiktok:@{}", set()),
    (re.compile(r"(?:x|twitter)\.com/([^/?#]+)", re.I), "x:@{}", {"search", "hashtag", "i", "home", "intent"}),
]


def normalize_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("595"):
        digits = "0" + digits[3:]
    return digits


def _unique(items) -> list:
    out = []
    for i in items:
        if i and i not in out:
            out.append(i)
    return out


def extract_entities(text: str, url: str = "") -> dict:
    """Phones, seller handles, prices, locations and vehicle attributes from public snippet text."""
    text = text or ""
    flat = strip_accents(text).lower()
    sellers = [f"@{h}" for h in HANDLE_RX.findall(text)]
    for rx, fmt, skip in _URL_USER_RULES:
        m = rx.search(url or "")
        if m and m.group(1).lower() not in skip:
            sellers.append(fmt.format(m.group(1)))
            break
    return {
        "phones": _unique(normalize_phone(p) for p in PHONE_RX.findall(text)),
        "sellers": _unique(sellers),
        "prices": _unique(re.sub(r"\s+", " ", p).strip(" .,") for p in PRICE_RX.findall(text)),
        "locations": _unique(_CANON[m] for m in _LOC_RX.findall(flat)),
        "plates": find_plates(text),
        "attributes": {
            "makes": _unique(_CANON[m] for m in _MAKE_RX.findall(flat)),
            "colors": _unique(_CANON[m] for m in _COLOR_RX.findall(flat)),
            "years": _unique(YEAR_RX.findall(text)),
        },
    }


# --------------------------------------------------------------------------- #
# Platform classification
# --------------------------------------------------------------------------- #
CLASSIFIED_HOSTS = ("clasipar.paraguay.com", "hendyla.com", "mercadolibre.com", "olx.com", "infoauto.com.py",
                    "autos.com.py", "tucarro", "encuentra24", "clasificados")
NEWS_HOSTS = ("abc.com.py", "ultimahora.com", "lanacion.com.py", "hoy.com.py", "extra.com.py", "npy.com.py",
              "cronica.com.py", "adndigital.com.py", "rdn.com.py", "elnacional.com.py", "paraguay.com",
              "ip.gov.py", "ministeriopublico.gov.py", "policianacional.gov.py")
SOCIAL_HOSTS = ("facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com", "youtube.com", "threads.net")
PLATFORMS = ["Marketplace", "Classifieds", "News", "Social Post", "Web"]


def classify_platform(url: str) -> str:
    p = urlparse(url if "://" in url else "https://" + url)
    host, path = p.netloc.lower().removeprefix("www.").removeprefix("m."), p.path.lower()
    if host.endswith("facebook.com") and path.startswith("/marketplace"):
        return "Marketplace"
    if any(h in host for h in CLASSIFIED_HOSTS):
        return "Classifieds"
    if any(host == h or host.endswith("." + h) for h in SOCIAL_HOSTS):
        return "Social Post"
    if any(host == h or host.endswith("." + h) for h in NEWS_HOSTS) or "/noticia" in path or "/nacionales" in path:
        return "News"
    return "Web"


# --------------------------------------------------------------------------- #
# Dates (for the timeline)
# --------------------------------------------------------------------------- #
_MONTHS = {"jan": 1, "ene": 1, "feb": 2, "mar": 3, "apr": 4, "abr": 4, "may": 5, "jun": 6, "jul": 7,
           "aug": 8, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dec": 12, "dic": 12}
_D_DMY = re.compile(r"(?<!\d)(\d{1,2})\s+(?:de\s+)?([a-z]{3})[a-z]*\.?,?\s+(?:de\s+)?(\d{4})(?!\d)")
_D_MDY = re.compile(r"(?<![a-z])([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})(?!\d)")
_D_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})")
_D_SLASH = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)")
_D_REL = re.compile(r"(?:hace\s+(\d+)\s+([a-zí]+)|(\d+)\s+([a-z]+)\s+ago)")
_REL_DAYS = {"hour": 0, "hora": 0, "day": 1, "dia": 1, "week": 7, "semana": 7, "month": 30, "mes": 30,
             "year": 365, "ano": 365}
_META_DATE_KEYS = ("article:published_time", "og:published_time", "datepublished", "date", "pubdate",
                   "og:updated_time", "article:modified_time")


def _mk(y: int, mo: int, d: int) -> datetime | None:
    try:
        dt = datetime(y, mo, d)
    except ValueError:
        return None
    return dt if 2000 <= y and dt <= datetime.now() + timedelta(days=2) else None


def extract_date(text: str, pagemap: dict | None = None, now: datetime | None = None) -> datetime | None:
    """Best-effort publication date from page metadata, then from the snippet text."""
    for tags in (pagemap or {}).get("metatags", []) or []:
        for key in _META_DATE_KEYS:
            m = _D_ISO.search(str(tags.get(key, "")))
            if m and (dt := _mk(int(m[1]), int(m[2]), int(m[3]))):
                return dt
    flat = strip_accents(text or "").lower()
    now = now or datetime.now()
    if m := _D_REL.search(flat):
        n, unit = (m[1], m[2]) if m[1] else (m[3], m[4])
        unit = unit.rstrip("s") if unit not in ("mes",) else unit
        unit = "mes" if unit in ("mese",) else unit
        if unit in _REL_DAYS:
            return (now - timedelta(days=int(n) * _REL_DAYS[unit])).replace(hour=0, minute=0, second=0, microsecond=0)
    if m := _D_ISO.search(flat):
        if dt := _mk(int(m[1]), int(m[2]), int(m[3])):
            return dt
    if (m := _D_DMY.search(flat)) and m[2] in _MONTHS:
        if dt := _mk(int(m[3]), _MONTHS[m[2]], int(m[1])):
            return dt
    if (m := _D_MDY.search(flat)) and m[1] in _MONTHS:
        if dt := _mk(int(m[3]), _MONTHS[m[1]], int(m[2])):
            return dt
    if m := _D_SLASH.search(flat):
        if dt := _mk(int(m[3]), int(m[2]), int(m[1])):  # dd/mm/yyyy (regional convention)
            return dt
    return None


# --------------------------------------------------------------------------- #
# Scoring a single hit
# --------------------------------------------------------------------------- #
def url_hash(url: str) -> str:
    cleaned = re.sub(r"[?#].*$", "", url.strip().lower()).rstrip("/")
    return hashlib.sha1(cleaned.encode("utf-8")).hexdigest()


def analyze_hit(params: dict, title: str, snippet: str, url: str, pagemap: dict | None = None) -> dict:
    """Turn one search hit into the fields stored on a Result row.

    Confidence scale:
      exact  1.00  the plate text appears verbatim (any spacing/hyphen variant)
      fuzzy  0.70-0.80  one character off, or an OCR-confusable variant (0/O, 1/I ...)
      context <=0.45  no plate in the text; only vehicle attributes matched
    """
    meta_desc = " ".join(str(t.get("og:description", "")) for t in (pagemap or {}).get("metatags", []) or [])
    text = " \n ".join(x for x in (title, snippet, meta_desc) if x)
    ents = extract_entities(text, url)

    flat = strip_accents(text).lower()
    wanted = [strip_accents(str(params.get(k) or "")).lower().strip() for k in ("make", "model", "year", "color", "location")]
    wanted = [w for w in wanted if w]
    hits = sum(1 for w in wanted if w in flat)

    mtype, score, matched = match_plate(params.get("plate", ""), text)
    if mtype == "exact":
        conf = 1.0
    elif mtype == "fuzzy":
        conf = round(score / 100 * 0.85, 2)
    else:
        mtype = "context"
        conf = round(0.15 + 0.30 * (hits / len(wanted)), 2) if wanted else 0.10

    return {
        "url": url, "url_hash": url_hash(url), "title": title or "", "snippet": snippet or "",
        "platform": classify_platform(url), "match_type": mtype, "matched_text": matched, "confidence": conf,
        "plates_found": ents["plates"], "phones": ents["phones"], "sellers": ents["sellers"],
        "prices": ents["prices"], "locations": ents["locations"], "attributes": ents["attributes"],
        "published_at": extract_date(text, pagemap),
    }


# --------------------------------------------------------------------------- #
# Manual import (Link mode): paste text copied from a results page or a post
# --------------------------------------------------------------------------- #
_URL_RX = re.compile(r"https?://[^\s<>\"')]+")


def parse_pasted(text: str) -> list[dict]:
    """Split pasted text into hits. Blank lines separate entries; a URL anchors each one."""
    hits = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        block = block.strip()
        if not block:
            continue
        urls = _URL_RX.findall(block)
        body = _URL_RX.sub("", block).strip()
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        url = urls[0].rstrip(".,") if urls else "manual://" + hashlib.sha1(block.encode("utf-8")).hexdigest()[:12]
        hits.append({"url": url, "title": (lines[0] if lines else url)[:200], "snippet": " ".join(lines[1:] or lines)[:1000]})
    return hits


# --------------------------------------------------------------------------- #
# Analytics across a result set
# --------------------------------------------------------------------------- #
def build_analytics(results: list[dict]) -> dict:
    """Recurring entities and a chronological timeline for a list of Result dicts."""
    buckets: dict[str, dict[str, list[int]]] = {k: defaultdict(list) for k in ("phones", "sellers", "prices", "locations", "plates_found")}
    for r in results:
        for kind, bucket in buckets.items():
            for value in r.get(kind) or []:
                bucket[value].append(r["id"])

    def ranked(bucket: dict[str, list[int]]) -> list[dict]:
        rows = [{"value": v, "count": len(ids), "result_ids": ids, "recurring": len(ids) > 1} for v, ids in bucket.items()]
        return sorted(rows, key=lambda x: (-x["count"], x["value"]))

    timeline = []
    for r in results:
        when = r.get("published_at") or r.get("first_seen")
        if when:
            timeline.append({
                "date": when[:10], "dated": bool(r.get("published_at")), "id": r["id"], "title": r["title"],
                "url": r["url"], "platform": r["platform"], "match_type": r["match_type"],
                "confidence": r["confidence"], "locations": r.get("locations") or [],
            })
    timeline.sort(key=lambda x: (x["date"], x["id"]))

    by_platform, by_match = defaultdict(int), defaultdict(int)
    for r in results:
        by_platform[r["platform"]] += 1
        by_match[r["match_type"]] += 1

    return {
        "total": len(results),
        "by_platform": dict(by_platform), "by_match": dict(by_match),
        "phones": ranked(buckets["phones"]), "sellers": ranked(buckets["sellers"]),
        "prices": ranked(buckets["prices"]), "locations": ranked(buckets["locations"]),
        "plates": ranked(buckets["plates_found"]), "timeline": timeline,
    }
