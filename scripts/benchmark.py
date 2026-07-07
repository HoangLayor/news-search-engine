"""So sánh hiệu quả & tốc độ 3 chế độ tìm kiếm: lexical vs dense vs hybrid.

- HIỆU QUẢ: đo ở tầng TRUY HỒI (trước decay/dedup) trên một bộ nhãn nhỏ (qrels)
  — Recall@10, MRR, nDCG@10 — để tách bạch chất lượng từng nhánh.
- TỐC ĐỘ: đo ở tầng PIPELINE end-to-end (search hoàn chỉnh) — p50/p95 mỗi truy vấn.

Chạy:  python benchmark.py         (mặc định EMBEDDER=hash offline)
       EMBEDDER=bge python benchmark.py   (nếu đã cài BGE-m3 -> dense mạnh hơn)

Lưu ý: với embedder 'hash' (char n-gram offline), nhánh dense yếu hơn lexical;
đó là điều benchmark này cho thấy. Production dùng BGE-m3/OpenAI -> dense cải thiện.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Cho phép chạy từ bất kỳ đâu và import khi ở trong package `scripts`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery
from news_search.search.fusion import rrf
from news_search.search.pipeline import SearchPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone(timedelta(hours=7)))
DATA = ROOT / "data" / "sample_articles.json"
K = 10

# Bộ nhãn (qrels): truy vấn -> tập bài viết được coi là LIÊN QUAN.
QRELS: dict[str, set[str]] = {
    "lạm phát tháng 6": {"eco-cpi-01", "eco-cpi-02"},
    "vì sao giá vàng tăng mạnh": {"eco-gold-01", "eco-gold-02"},
    "giá xăng điều chỉnh": {"eco-gas-01", "soc-gas-02"},
    "động đất Kon Tum": {"news-quake-new", "news-quake-old"},
    "kết quả bầu cử tổng thống": {"elect-dup-1", "elect-dup-2", "elect-dup-3", "elect-analysis"},
    "đội tuyển bóng đá quốc gia": {"sport-01", "sport-02"},
    "trí tuệ nhân tạo báo chí": {"tech-01"},
    "tuyển sinh đại học": {"edu-02", "edu-01"},
    "lãi suất ngân hàng nhà nước": {"eco-bank-01"},
    "sốt xuất huyết mùa mưa": {"health-01"},
}


# ------------------------------------------------------------------ metrics


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for aid in ranked[:k] if aid in relevant)
    return hit / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, aid in enumerate(ranked, start=1):
        if aid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, aid in enumerate(ranked[:k], start=1)
        if aid in relevant
    )
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


# ------------------------------------------------------------- retrieval per mode


def retrieve(manager: IndexManager, text: str, mode: str, k: int) -> list[str]:
    """Trả danh sách id ứng viên theo từng nhánh (KHÔNG decay/dedup)."""
    lex = manager.lexical.search(text, top_k=k)
    if mode == "lexical":
        return [a for a, _ in lex]
    qvec = manager.embedder.embed([text])[0]
    vec = manager.vector.search(qvec, top_k=k)
    if mode == "dense":
        return [a for a, _ in vec]
    if mode == "hybrid":
        fused = rrf([[a for a, _ in lex], [a for a, _ in vec]], k=manager.settings.rrf_k)
        return [a for a, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    raise ValueError(mode)


# ----------------------------------------------------------------------- main


def main() -> None:
    # Benchmark trên 29 bài -> luôn dùng vector index local (brute-force); embedder
    # lấy theo cấu hình (.env), vd EMBEDDER=bge để so BGE-m3, EMBEDDER=hash cho nhanh.
    settings = Settings(vector_backend="local")
    manager = IndexManager(settings)
    manager.bulk_index(json.loads(DATA.read_text(encoding="utf-8")))
    pipeline = SearchPipeline(manager, settings)
    modes = ["lexical", "dense", "hybrid"]

    print(f"Embedder = {settings.embedder} (dim={manager.embedder.dim}) · "
          f"vector_backend = {settings.vector_backend} · {len(QRELS)} truy vấn nhãn\n")

    # --- Hiệu quả (retrieval) ---
    print("HIỆU QUẢ (tầng truy hồi, trung bình trên bộ qrels)")
    print(f"{'mode':<9}{'Recall@'+str(K):>12}{'MRR':>10}{'nDCG@'+str(K):>12}")
    eff: dict[str, dict[str, float]] = {}
    for mode in modes:
        rs, ms, ns = [], [], []
        for q, rel in QRELS.items():
            ranked = retrieve(manager, q, mode, K)
            rs.append(recall_at_k(ranked, rel, K))
            ms.append(mrr(ranked, rel))
            ns.append(ndcg_at_k(ranked, rel, K))
        eff[mode] = {
            "recall": statistics.mean(rs),
            "mrr": statistics.mean(ms),
            "ndcg": statistics.mean(ns),
        }
        print(f"{mode:<9}{eff[mode]['recall']:>12.3f}{eff[mode]['mrr']:>10.3f}{eff[mode]['ndcg']:>12.3f}")

    # --- Tốc độ (pipeline end-to-end) ---
    print("\nTỐC ĐỘ (pipeline end-to-end, ms/truy vấn)")
    print(f"{'mode':<9}{'p50':>10}{'p95':>10}{'mean':>10}")
    # Pipeline dùng tên "semantic" cho nhánh dense
    pipe_mode = {"lexical": "lexical", "dense": "semantic", "hybrid": "hybrid"}
    reps = 30
    for mode in modes:
        lat: list[float] = []
        for _ in range(reps):
            for q in QRELS:
                t0 = time.perf_counter()
                pipeline.search(SearchQuery(text=q, mode=pipe_mode[mode], top_k=K, now=NOW))
                lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        p50 = lat[int(0.50 * len(lat))]
        p95 = lat[int(0.95 * len(lat))]
        print(f"{mode:<9}{p50:>10.2f}{p95:>10.2f}{statistics.mean(lat):>10.2f}")

    # --- Kết luận tự động ---
    best = max(eff, key=lambda m: eff[m]["ndcg"])
    print(f"\nnDCG@{K} cao nhất: {best} ({eff[best]['ndcg']:.3f}). "
          "Hybrid thường cân bằng tốt recall (lexical) + ngữ nghĩa (dense).")


if __name__ == "__main__":
    main()
