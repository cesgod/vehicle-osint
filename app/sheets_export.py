"""Exports: Google Sheets report in the user's Drive (Advanced mode) and CSV (any mode).

Quota note: Sheets API allows 60 write requests/minute per user. One export uses
3 Sheets calls (create, values batch, formatting batch) plus one Drive upload per
plate crop, capped at MAX_CROPS.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime

from . import config, google_auth
from .analyzer import build_analytics, format_plate
from .models import ImageScan, Search

MAX_CROPS = 25
RESULT_HEADER = ["Confidence", "Match", "Matched text", "Platform", "Title", "URL", "Published", "First seen",
                 "Plates in text", "Phones", "Sellers", "Prices", "Locations", "Snippet"]


def _result_rows(results: list[dict]) -> list[list]:
    j = lambda xs: ", ".join(xs or [])  # noqa: E731
    return [[r["confidence"], r["match_type"], r["matched_text"], r["platform"], r["title"], r["url"],
             (r["published_at"] or "")[:10], (r["first_seen"] or "")[:10], j(r["plates_found"]), j(r["phones"]),
             j(r["sellers"]), j(r["prices"]), j(r["locations"]), r["snippet"]]
            for r in sorted(results, key=lambda x: -x["confidence"])]


def results_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(RESULT_HEADER)
    writer.writerows(_result_rows(results))
    return buf.getvalue()


def _upload_crops(scans: list[ImageScan], title: str) -> dict[int, str]:
    """Upload plate crops to a Drive folder; returns scan id -> link."""
    from googleapiclient.http import MediaFileUpload

    with_crops = [s for s in scans if s.crop_file and (config.CROPS_DIR / s.crop_file).exists()][:MAX_CROPS]
    if not with_crops:
        return {}
    drive = google_auth.drive_service()
    folder = drive.files().create(body={"name": f"{title} - plate evidence",
                                        "mimeType": "application/vnd.google-apps.folder"}, fields="id").execute()
    links = {}
    for s in with_crops:
        media = MediaFileUpload(str(config.CROPS_DIR / s.crop_file), mimetype="image/jpeg")
        f = drive.files().create(body={"name": f"{s.best_plate or 'region'}_{s.filename}.jpg", "parents": [folder["id"]]},
                                 media_body=media, fields="id,webViewLink").execute()
        links[s.id] = f.get("webViewLink", "")
    return links


def export_to_sheets(db, search: Search) -> str:
    """Create a formatted spreadsheet report. Returns its URL."""
    results = [r.to_dict() for r in search.results]
    analytics = build_analytics(results)
    scans = db.query(ImageScan).order_by(ImageScan.id.desc()).limit(200).all()
    subject = format_plate(search.plate) or " ".join(x for x in (search.make, search.model) if x) or "vehicle"
    title = f"Vehicle OSINT - {subject} - {datetime.now():%Y-%m-%d %H.%M}"

    crop_links = _upload_crops(scans, title)

    entity_rows = [["Type", "Value", "Appearances", "Recurring"]]
    for kind in ("phones", "sellers", "prices", "locations", "plates"):
        entity_rows += [[kind, e["value"], e["count"], "yes" if e["recurring"] else ""] for e in analytics[kind]]
    scan_rows = [["File", "Engine", "Best plate", "Confidence", "All plates", "Vehicle colours", "Cross-references", "Plate crop"]]
    for s in scans:
        scan_rows.append([s.filename, s.engine, s.best_plate, round(s.confidence or 0, 2),
                          ", ".join(p["plate"] for p in s.plates or []), ", ".join(s.colors or []),
                          len(s.matches or []), crop_links.get(s.id, "")])

    tabs = [("Results", [RESULT_HEADER] + _result_rows(results)), ("Entities", entity_rows), ("Image scans", scan_rows)]
    sheets = google_auth.sheets_service()
    book = sheets.spreadsheets().create(body={
        "properties": {"title": title},
        "sheets": [{"properties": {"sheetId": i, "title": name, "gridProperties": {"frozenRowCount": 1}}}
                   for i, (name, _rows) in enumerate(tabs)],
    }, fields="spreadsheetId,spreadsheetUrl").execute()

    sheets.spreadsheets().values().batchUpdate(spreadsheetId=book["spreadsheetId"], body={
        "valueInputOption": "RAW",
        "data": [{"range": f"'{name}'!A1", "values": rows} for name, rows in tabs],
    }).execute()

    requests = []
    for i, (_name, rows) in enumerate(tabs):
        requests += [
            {"repeatCell": {"range": {"sheetId": i, "startRowIndex": 0, "endRowIndex": 1},
                            "cell": {"userEnteredFormat": {
                                "backgroundColor": {"red": 0.06, "green": 0.2, "blue": 0.45},
                                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}}}},
                            "fields": "userEnteredFormat(backgroundColor,textFormat)"}},
            {"autoResizeDimensions": {"dimensions": {"sheetId": i, "dimension": "COLUMNS", "startIndex": 0,
                                                     "endIndex": len(rows[0])}}},
            {"setBasicFilter": {"filter": {"range": {"sheetId": i, "startRowIndex": 0, "endRowIndex": max(1, len(rows)),
                                                     "startColumnIndex": 0, "endColumnIndex": len(rows[0])}}}},
        ]
    # Confidence column as a percentage, with a green -> red scale.
    requests += [
        {"repeatCell": {"range": {"sheetId": 0, "startRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": 1},
                        "cell": {"userEnteredFormat": {"numberFormat": {"type": "PERCENT", "pattern": "0%"}}},
                        "fields": "userEnteredFormat.numberFormat"}},
        {"addConditionalFormatRule": {"index": 0, "rule": {
            "ranges": [{"sheetId": 0, "startRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": 1}],
            "gradientRule": {"minpoint": {"color": {"red": 0.96, "green": 0.8, "blue": 0.8}, "type": "NUMBER", "value": "0"},
                             "maxpoint": {"color": {"red": 0.72, "green": 0.88, "blue": 0.75}, "type": "NUMBER", "value": "1"}}}}},
    ]
    sheets.spreadsheets().batchUpdate(spreadsheetId=book["spreadsheetId"], body={"requests": requests}).execute()
    return book["spreadsheetUrl"]
