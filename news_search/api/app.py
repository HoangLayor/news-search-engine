"""HTTP API (FastAPI) — CONTRACTS.md §16 + tính năng vận hành GĐ5 (bật/tắt được).

- /search luôn trả JSON array, mỗi phần tử ĐÚNG 5 trường (contract giữ nguyên).
- Feedback (GĐ1): khi FEEDBACK_ENABLED, tự log lượt tìm + trả header X-Search-Id;
  client gọi /events/click để log click/dwell.
- Metrics/dashboard/alert (GĐ5): /metrics (Prometheus), /dashboard (HTML).
- Blue-green reindex (GĐ5): /admin/reindex (bảo vệ bằng ADMIN_TOKEN).
- A/B (GĐ5): khi EXPERIMENT_ENABLED, gán biến thể (X-Variant) + log để phân tích.
Mọi tính năng phụ mặc định tắt/an toàn; thiếu phụ thuộc thì degrade, không crash.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

from fastapi import Body, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

from news_search.config import Settings
from news_search.feedback.base import ClickEvent, SearchEvent
from news_search.ingest.cleaner import TZ_VN
from news_search.models import SearchFilters, SearchQuery
from news_search.service.metrics import render_dashboard
from news_search.service.search_service import SearchService

_UI_HTML_PATH = Path(__file__).parent / "ui.html"


def _feedback_stats(settings: Settings) -> Optional[dict]:
    """CTR/zero-result từ log feedback (chỉ backend jsonl)."""
    if settings.feedback_enabled and settings.feedback_backend == "jsonl":
        from news_search.feedback.analytics import compute_metrics

        return compute_metrics(settings.feedback_path)
    return None


def _parse_dt(value: Optional[str], field: str) -> Optional[datetime]:
    """Parse tham số ngày ISO-8601; naive -> gán UTC+7. Lỗi -> HTTP 400."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} không phải ISO-8601: {value!r}") from exc
    return dt if dt.tzinfo else dt.replace(tzinfo=TZ_VN)


def _variant(q: str) -> str:
    """Bucket A/B deterministic theo truy vấn (khung A/B; log để phân tích online)."""
    return "B" if int(hashlib.md5(q.encode("utf-8")).hexdigest(), 16) % 2 else "A"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Tạo FastAPI app; dùng SearchService (hỗ trợ reindex blue-green + metrics + feedback)."""
    settings = settings or Settings.from_env()
    svc = SearchService(settings)

    app = FastAPI(title="News Search Engine", version="0.2.0")
    app.state.service = svc

    # Tự nạp bài mẫu lúc khởi động (tiện demo Docker): chỉ khi bật cờ, nguồn sample
    # và chỉ mục đang rỗng — an toàn, không lặp.
    if settings.autoload_sample and settings.source == "sample" and len(svc.manager.store) == 0:
        try:
            from news_search.sources import get_source

            src = get_source(settings)
        except Exception:  # không mở được nguồn -> app vẫn khởi động (chỉ mục rỗng)
            _log.exception("Autoload: không mở được nguồn mẫu")
        else:
            try:
                # Nạp resilient TỪNG bài: 1 bài lỗi không làm hỏng cả đợt (khác bulk_index)
                for raw in src.fetch_all(settings.ingest_batch_size):
                    try:
                        svc.manager.index_article(raw)
                    except Exception:
                        _log.exception("Autoload: bỏ qua 1 bài lỗi")
            finally:
                src.close()

    # ---------------------------------------------------------------- ingest

    @app.post("/articles", status_code=201)
    def index_article(raw: dict = Body(...)) -> dict:
        try:
            article = svc.manager.index_article(raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"indexed": article.status == "published", "article_id": article.article_id}

    @app.post("/articles/bulk")
    def index_bulk(raws: list[dict] = Body(...)) -> dict:
        try:
            n = svc.manager.bulk_index(raws)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"indexed": n}

    @app.delete("/articles/{article_id}")
    def delete_article(article_id: str) -> dict:
        svc.manager.remove_article(article_id)
        return {"removed": True}

    @app.get("/articles/{article_id}/entities")
    def article_entities(article_id: str) -> list[dict]:
        if svc.manager.store.get(article_id) is None:
            raise HTTPException(status_code=404, detail=f"Không có bài {article_id!r}")
        return [{"name": e.name, "type": e.type}
                for e in svc.manager.kg.entities_for_article(article_id)]

    # ---------------------------------------------------------------- search

    @app.get("/search")
    def search(
        response: Response,
        q: str = Query("", description="Truy vấn (từ khóa hoặc câu hỏi)"),
        mode: str = Query("hybrid"),
        top_k: int = Query(10, ge=1, le=100),
        author: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        source: Optional[str] = Query(None),
        date_from: Optional[str] = Query(None),
        date_to: Optional[str] = Query(None),
    ) -> list[dict]:
        """Trả JSON array các bài viết xếp hạng — mỗi phần tử ĐÚNG 5 trường."""
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="Thiếu truy vấn 'q'")
        if mode not in ("lexical", "semantic", "hybrid"):
            raise HTTPException(status_code=400, detail=f"mode không hợp lệ: {mode!r}")
        filters = SearchFilters(
            author=author, category=category, source=source,
            date_from=_parse_dt(date_from, "date_from"),
            date_to=_parse_dt(date_to, "date_to"),
        )
        query = SearchQuery(text=q, filters=filters, top_k=top_k, mode=mode)
        t0 = time.perf_counter()
        try:
            results = svc.pipeline.search(query)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        items = [item.to_dict() for item in results]

        variant = _variant(q) if settings.experiment_enabled else None
        if svc.metrics is not None:
            svc.metrics.record_search(latency_ms, len(items))
        if settings.feedback_enabled:
            sid = uuid.uuid4().hex
            svc.events.log_search(SearchEvent(
                search_id=sid, query=q, mode=mode, top_k=top_k,
                result_ids=[it["article_id"] for it in items],
                num_results=len(items), zero_result=(len(items) == 0), variant=variant,
            ))
            response.headers["X-Search-Id"] = sid  # để client gắn click về lượt này
        if variant:
            response.headers["X-Variant"] = variant
        return items

    @app.get("/explain")
    def explain(
        q: str = Query("", description="Truy vấn cần giải thích"),
        mode: str = Query("hybrid"),
        top_k: int = Query(10, ge=1, le=100),
        author: Optional[str] = Query(None),
        category: Optional[str] = Query(None),
        source: Optional[str] = Query(None),
        date_from: Optional[str] = Query(None),
        date_to: Optional[str] = Query(None),
    ) -> dict:
        """Giải thích pipeline: trả {query, mode, results, trace} — từng bước xử lý
        + kết quả trung gian (bỏ qua cache). Phục vụ UI quan sát cách xếp hạng."""
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="Thiếu truy vấn 'q'")
        if mode not in ("lexical", "semantic", "hybrid"):
            raise HTTPException(status_code=400, detail=f"mode không hợp lệ: {mode!r}")
        filters = SearchFilters(
            author=author, category=category, source=source,
            date_from=_parse_dt(date_from, "date_from"),
            date_to=_parse_dt(date_to, "date_to"),
        )
        query = SearchQuery(text=q, filters=filters, top_k=top_k, mode=mode)
        try:
            return svc.pipeline.explain(query)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------- feedback (GĐ1)

    @app.post("/events/click")
    def log_click(payload: dict = Body(...)) -> dict:
        """Log click/dwell. Cần search_id, article_id, position (dwell_ms tùy chọn).

        FEEDBACK_ENABLED=false -> logger no-op (trả logged=false), không lỗi.
        """
        try:
            evt = ClickEvent(
                search_id=str(payload["search_id"]),
                article_id=str(payload["article_id"]),
                position=int(payload["position"]),
                dwell_ms=int(payload["dwell_ms"]) if payload.get("dwell_ms") is not None else None,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=f"payload click không hợp lệ: {exc}") from exc
        svc.events.log_click(evt)
        return {"logged": settings.feedback_enabled}

    # ------------------------------------------------------------- ops (GĐ5)

    @app.get("/metrics")
    def metrics() -> Response:
        """Số liệu Prometheus text. METRICS_ENABLED=false -> 404."""
        if svc.metrics is None:
            raise HTTPException(status_code=404, detail="metrics đã tắt")
        return PlainTextResponse(svc.metrics.prometheus_text())

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        """Trang dashboard HTML. DASHBOARD_ENABLED=false -> 404."""
        if not settings.dashboard_enabled:
            raise HTTPException(status_code=404, detail="dashboard đã tắt")
        metrics_data = svc.metrics.snapshot() if svc.metrics else {}
        feedback_data = None
        if settings.feedback_enabled and settings.feedback_backend == "jsonl":
            from news_search.feedback.analytics import compute_metrics
            feedback_data = compute_metrics(settings.feedback_path)
        counts = {
            "articles": len(svc.manager.store),
            "generation": svc.manager.generation,
            "embedder": settings.embedder,
            "vector_backend": settings.vector_backend,
        }
        # Kiểm ngưỡng cảnh báo -> nhúng vào dashboard
        alerts = svc.metrics.check_alerts(
            settings.alert_p95_ms, settings.alert_error_rate, settings.alert_zero_result_rate
        ) if svc.metrics else []
        if alerts:
            metrics_data = {**metrics_data, "ALERTS": "; ".join(alerts)}
        return HTMLResponse(render_dashboard(metrics_data, feedback_data, counts))

    @app.post("/admin/reindex")
    def admin_reindex(
        payload: dict = Body(default={}),
        x_admin_token: str = Header(default=""),
    ) -> dict:
        """Reindex blue-green từ nguồn cấu hình. Tắt nếu ADMIN_TOKEN rỗng; sai token -> 403."""
        if not settings.admin_token:
            raise HTTPException(status_code=404, detail="admin đã tắt (đặt ADMIN_TOKEN để bật)")
        # So sánh constant-time -> tránh rò rỉ token qua kênh thời gian
        if not hmac.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(status_code=403, detail="token admin không đúng")
        limit = payload.get("limit")
        n = svc.reindex(limit=int(limit) if limit else None)
        return {"reindexed": n, "generation": svc.manager.generation}

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "articles": len(svc.manager.store),
            "lexical": len(svc.manager.lexical),
            "vectors": len(svc.manager.vector),
            "generation": svc.manager.generation,
            "embedder": settings.embedder,
            "vector_backend": settings.vector_backend,
            "features": {
                "feedback": settings.feedback_enabled,
                "cache": settings.cache_enabled,
                "metrics": settings.metrics_enabled,
                "experiment": settings.experiment_enabled,
                "qu": svc.pipeline.understander.enabled,
            },
        }

    # ------------------------------------------------------------- Demo UI

    @app.get("/stats")
    def stats() -> dict:
        """Số liệu tổng hợp cho UI (counts + metrics + feedback CTR)."""
        return {
            "counts": {
                "articles": len(svc.manager.store),
                "generation": svc.manager.generation,
            },
            "backends": {"embedder": settings.embedder, "vector": settings.vector_backend,
                         "lexical": settings.lexical_backend, "reranker": settings.reranker},
            "features": {
                "feedback": settings.feedback_enabled, "cache": settings.cache_enabled,
                "metrics": settings.metrics_enabled, "experiment": settings.experiment_enabled,
                "qu": svc.pipeline.understander.enabled, "admin": bool(settings.admin_token),
            },
            "metrics": svc.metrics.snapshot() if svc.metrics else None,
            "feedback": _feedback_stats(settings),
        }

    @app.get("/", response_class=HTMLResponse)
    @app.get("/ui", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        """Trang demo UI tương tác. UI_ENABLED=false -> 404."""
        if not settings.ui_enabled:
            raise HTTPException(status_code=404, detail="UI đã tắt (đặt UI_ENABLED=true)")
        try:
            html = _UI_HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:  # pragma: no cover
            raise HTTPException(status_code=500, detail="Thiếu ui.html")
        return HTMLResponse(html)

    return app


# `app` được tạo LAZY (PEP 562): chỉ dựng khi truy cập thuộc tính `app`
# (vd uvicorn `news_search.api.app:app`). Nhờ vậy `import create_app` / import module
# KHÔNG kích hoạt kết nối Milvus/tải BGE theo .env — quan trọng cho test & notebook.
_app = None


def __getattr__(name: str):
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
