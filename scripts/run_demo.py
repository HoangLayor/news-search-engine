"""Demo end-to-end: nạp bài mẫu rồi chạy 5 truy vấn ví dụ (BaoCao-ThietKe.md §3).

Thời điểm "hiện tại" cố định (2026-07-07 12:00 +07:00) để output ổn định giữa
các lần chạy. Chạy:  python run_demo.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Cho phép chạy `python scripts/run_demo.py` từ bất kỳ đâu: thêm gốc dự án vào path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchFilters, SearchQuery
from news_search.search.pipeline import SearchPipeline

# Windows console -> UTF-8 để in tiếng Việt không lỗi
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone(timedelta(hours=7)))
DATA = ROOT / "data" / "sample_articles.json"


def _run(pipeline: SearchPipeline, label: str, query: SearchQuery) -> None:
    results = pipeline.search(query)
    print("\n" + "=" * 70)
    print(f"TRUY VẤN: {label}")
    print("=" * 70)
    print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))


def main() -> None:
    # Demo offline & deterministic: ép local/hash bất kể .env (production dùng .env)
    settings = Settings(vector_backend="local", embedder="hash")
    manager = IndexManager(settings)
    raws = json.loads(DATA.read_text(encoding="utf-8"))
    n = manager.bulk_index(raws)
    pipeline = SearchPipeline(manager, settings)
    print(f"Đã index {n} bài viết. Chạy {5} truy vấn ví dụ (now = {NOW.isoformat()}):")

    _run(pipeline, 'Từ khóa: "lạm phát tháng 6"',
         SearchQuery(text="lạm phát tháng 6", top_k=5, now=NOW))

    _run(pipeline, 'Câu hỏi tự nhiên: "vì sao giá vàng tăng mạnh tuần này"',
         SearchQuery(text="vì sao giá vàng tăng mạnh tuần này", top_k=5, now=NOW))

    _run(pipeline, 'Tin nóng: "động đất"',
         SearchQuery(text="động đất", top_k=5, now=NOW))

    _run(pipeline, 'Từ khóa + lọc chuyên mục Kinh tế: "giá xăng"',
         SearchQuery(text="giá xăng", top_k=5, now=NOW,
                     filters=SearchFilters(category="Kinh tế")))

    _run(pipeline, 'Sự kiện nhiều bài trùng: "kết quả bầu cử"',
         SearchQuery(text="kết quả bầu cử", top_k=10, now=NOW))


if __name__ == "__main__":
    main()
