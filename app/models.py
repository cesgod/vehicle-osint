"""SQLite models: search history, deduplicated results, image scans, API cache and quota log."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


class Base(DeclarativeBase):
    pass


class Search(Base):
    __tablename__ = "searches"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate: Mapped[str] = mapped_column(String(16), default="", index=True)  # normalized, e.g. ABCD123
    make: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    year: Mapped[str] = mapped_column(String(8), default="")
    color: Mapped[str] = mapped_column(String(32), default="")
    location: Mapped[str] = mapped_column(String(64), default="")
    mode: Mapped[str] = mapped_column(String(16), default="link")  # api | link | demo
    queries: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    results: Mapped[list["Result"]] = relationship(back_populates="search", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "plate": self.plate, "make": self.make, "model": self.model,
            "year": self.year, "color": self.color, "location": self.location, "mode": self.mode,
            "queries": self.queries or [], "created_at": iso(self.created_at),
            "result_count": len(self.results),
        }


class Result(Base):
    __tablename__ = "results"
    # Deduplication: one row per URL per search.
    __table_args__ = (UniqueConstraint("search_id", "url_hash", name="uq_result_search_url"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    search_id: Mapped[int] = mapped_column(ForeignKey("searches.id", ondelete="CASCADE"), index=True)
    url: Mapped[str] = mapped_column(Text)
    url_hash: Mapped[str] = mapped_column(String(40), index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    snippet: Mapped[str] = mapped_column(Text, default="")
    display_link: Mapped[str] = mapped_column(String(255), default="")
    thumbnail: Mapped[str] = mapped_column(Text, default="")
    image_url: Mapped[str] = mapped_column(Text, default="")
    platform: Mapped[str] = mapped_column(String(32), default="Web", index=True)
    query_labels: Mapped[list] = mapped_column(JSON, default=list)
    match_type: Mapped[str] = mapped_column(String(16), default="context", index=True)  # exact|fuzzy|ocr|context
    matched_text: Mapped[str] = mapped_column(String(32), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plates_found: Mapped[list] = mapped_column(JSON, default=list)
    phones: Mapped[list] = mapped_column(JSON, default=list)
    sellers: Mapped[list] = mapped_column(JSON, default=list)
    prices: Mapped[list] = mapped_column(JSON, default=list)
    locations: Mapped[list] = mapped_column(JSON, default=list)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)  # makes / colors / years seen in text
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    search: Mapped[Search] = relationship(back_populates="results")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "search_id": self.search_id, "url": self.url, "title": self.title,
            "snippet": self.snippet, "display_link": self.display_link, "thumbnail": self.thumbnail,
            "image_url": self.image_url, "platform": self.platform, "query_labels": self.query_labels or [],
            "match_type": self.match_type, "matched_text": self.matched_text,
            "confidence": round(self.confidence or 0.0, 2), "plates_found": self.plates_found or [],
            "phones": self.phones or [], "sellers": self.sellers or [], "prices": self.prices or [],
            "locations": self.locations or [], "attributes": self.attributes or {},
            "published_at": iso(self.published_at), "first_seen": iso(self.first_seen),
        }


class ImageScan(Base):
    __tablename__ = "image_scans"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_path: Mapped[str] = mapped_column(Text, default="")
    engine: Mapped[str] = mapped_column(String(24), default="none")  # google-vision | easyocr | none
    raw_text: Mapped[str] = mapped_column(Text, default="")
    plates: Mapped[list] = mapped_column(JSON, default=list)  # [{plate, format, confidence, corrected}]
    best_plate: Mapped[str] = mapped_column(String(16), default="", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    vehicles: Mapped[list] = mapped_column(JSON, default=list)  # [{name, score, box}]
    colors: Mapped[list] = mapped_column(JSON, default=list)
    crop_file: Mapped[str] = mapped_column(String(255), default="")
    matches: Mapped[list] = mapped_column(JSON, default=list)  # cross-references into results/searches
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "filename": self.filename, "source_path": self.source_path,
            "engine": self.engine, "raw_text": self.raw_text, "plates": self.plates or [],
            "best_plate": self.best_plate, "confidence": round(self.confidence or 0.0, 2),
            "vehicles": self.vehicles or [], "colors": self.colors or [],
            "crop_url": f"/crops/{self.crop_file}" if self.crop_file else "",
            "matches": self.matches or [], "error": self.error, "created_at": iso(self.created_at),
        }


class QueryCache(Base):
    """Raw Custom Search responses, so repeating a query costs no quota."""
    __tablename__ = "query_cache"

    key: Mapped[str] = mapped_column(String(40), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    response: Mapped[dict] = mapped_column(JSON)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ApiUsage(Base):
    """Local quota counter. period = YYYY-MM-DD (Custom Search) or YYYY-MM (Vision)."""
    __tablename__ = "api_usage"
    __table_args__ = (UniqueConstraint("api", "period", name="uq_usage_api_period"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    api: Mapped[str] = mapped_column(String(24))
    period: Mapped[str] = mapped_column(String(10))
    count: Mapped[int] = mapped_column(Integer, default=0)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def usage(db, api: str, period: str) -> int:
    row = db.query(ApiUsage).filter_by(api=api, period=period).first()
    return row.count if row else 0


def add_usage(db, api: str, period: str, units: int = 1) -> None:
    row = db.query(ApiUsage).filter_by(api=api, period=period).first()
    if row is None:
        row = ApiUsage(api=api, period=period, count=0)
        db.add(row)
    row.count += units
    db.commit()
