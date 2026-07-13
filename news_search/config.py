"""Cấu hình hệ thống — chọn backend qua biến môi trường.

Mặc định mọi backend là "local" (pure-Python, chạy offline, không cần dịch vụ
ngoài) để dev/test/demo. Production đổi qua env:
    LEXICAL_BACKEND=opensearch, VECTOR_BACKEND=milvus, EMBEDDER=openai, RERANKER=bge
"""

from __future__ import annotations

import os

# Thiết lập giới hạn thread cho các thư viện tuyến tính để tránh deadlock OpenMP trong worker threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "disabled"


# Lấy đường dẫn gốc của dự án (news-search-engine)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thiết lập thư mục tải mô hình Hugging Face (BGE) mặc định vào thư mục .venv của dự án
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(BASE_DIR, ".venv", "huggingface_cache")

# Nạp biến môi trường từ .env ở gốc dự án (override=False -> env/shell luôn thắng
# .env; và các giá trị conftest/test set trước đó không bị .env đè). Không có
# python-dotenv thì bỏ qua (chỉ dùng os.environ).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)
except ImportError:  # pragma: no cover
    pass

from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Đọc cờ boolean: 1/true/yes/on -> True; 0/false/no/off -> False."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- Lựa chọn backend ---
    lexical_backend: str = field(default_factory=lambda: _env("LEXICAL_BACKEND", "local"))
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "local"))  # local | milvus
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "vi"))  # hash | vi | bge | openai
    reranker: str = field(default_factory=lambda: _env("RERANKER", "none"))  # none | vi | bge

    # --- Embedding ---
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 256))  # chỉ áp cho HashingEmbedder
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    )
    # BGE-m3 local (đa ngữ, hỗ trợ tiếng Việt) — dense 1024 chiều
    bge_model: str = field(default_factory=lambda: _env("BGE_MODEL", "BAAI/bge-m3"))
    bge_dim: int = field(default_factory=lambda: _env_int("BGE_DIM", 1024))
    bge_device: str = field(default_factory=lambda: _env("BGE_DEVICE", ""))  # "" -> tự chọn (cuda nếu có)
    # Embedder tiếng Việt CHUYÊN BIỆT (fine-tune BGE-m3) — chất lượng semantic VN cao hơn
    vi_embed_model: str = field(default_factory=lambda: _env("VI_EMBED_MODEL", "AITeamVN/Vietnamese_Embedding"))
    vi_embed_dim: int = field(default_factory=lambda: _env_int("VI_EMBED_DIM", 1024))
    # Reranker tiếng Việt chuyên biệt (cross-encoder, fine-tune bge-reranker-v2-m3)
    vi_rerank_model: str = field(default_factory=lambda: _env("VI_RERANK_MODEL", "AITeamVN/Vietnamese_Reranker"))

    # --- Tách từ (F-03) ---
    # auto: pyvi -> underthesea -> regex; hoặc ép: pyvi | underthesea | regex
    tokenizer: str = field(default_factory=lambda: _env("TOKENIZER", "auto"))

    # --- Milvus (VECTOR_BACKEND=milvus), chỉ mục HNSW ---
    milvus_uri: str = field(default_factory=lambda: _env("MILVUS_URI", "http://localhost:19530"))
    milvus_token: str = field(default_factory=lambda: _env("MILVUS_TOKEN", ""))
    milvus_collection: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "news_articles"))
    milvus_metric: str = field(default_factory=lambda: _env("MILVUS_METRIC", "COSINE"))
    # Collection đã tồn tại nhưng dim khác embedder -> mặc định BÁO LỖI rõ ràng.
    # Đặt true để TỰ ĐỘNG DROP + tạo lại (PHÁ HỦY dữ liệu vector cũ).
    milvus_recreate_on_dim_mismatch: bool = field(
        default_factory=lambda: _env_bool("MILVUS_RECREATE_ON_DIM_MISMATCH", False)
    )
    hnsw_m: int = field(default_factory=lambda: _env_int("HNSW_M", 16))
    hnsw_ef_construction: int = field(default_factory=lambda: _env_int("HNSW_EF_CONSTRUCTION", 200))
    hnsw_ef_search: int = field(default_factory=lambda: _env_int("HNSW_EF_SEARCH", 64))

    # --- Retrieval / fusion ---
    candidates_k: int = field(default_factory=lambda: _env_int("CANDIDATES_K", 100))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))

    # --- Time-decay (F-13, QDF) ---
    # Triết lý QDF: freshness chỉ DOMINATE khi truy vấn "đáng được làm mới"
    # (fresh_intent). Truy vấn thường -> decay NHẸ (relevance vẫn chủ đạo, bài
    # liên quan cũ hơn không bị vùi dưới bài mới yếu liên quan). Truy vấn tin
    # nóng -> half-life ngắn, recency áp đảo. floor giới hạn mức phạt tối đa để
    # điểm freshness không lấn át tín hiệu truy hồi (RRF nén biên độ điểm).
    half_life_days: float = field(default_factory=lambda: _env_float("HALF_LIFE_DAYS", 30.0))
    fresh_half_life_days: float = field(
        default_factory=lambda: _env_float("FRESH_HALF_LIFE_DAYS", 2.0)
    )
    decay_floor: float = field(default_factory=lambda: _env_float("DECAY_FLOOR", 0.5))

    # --- Rerank ---
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 50))

    # --- Dedup / đa dạng hóa (F-07, F-14) ---
    dedup_backend: str = field(default_factory=lambda: _env("DEDUP_BACKEND", "local"))  # local | datasketch
    dedup_threshold: float = field(default_factory=lambda: _env_float("DEDUP_THRESHOLD", 0.7))
    dedup_num_perm: int = field(default_factory=lambda: _env_int("DEDUP_NUM_PERM", 128))
    dedup_shingle_size: int = field(default_factory=lambda: _env_int("DEDUP_SHINGLE_SIZE", 3))
    max_per_cluster: int = field(default_factory=lambda: _env_int("MAX_PER_CLUSTER", 1))
    mmr_lambda: float = field(default_factory=lambda: _env_float("MMR_LAMBDA", 0.7))
    mmr_pool: int = field(default_factory=lambda: _env_int("MMR_POOL", 30))

    # --- Snippet (F-15) ---
    snippet_max_len: int = field(default_factory=lambda: _env_int("SNIPPET_MAX_LEN", 200))

    # --- Nguồn dữ liệu (data source) ---
    source: str = field(default_factory=lambda: _env("SOURCE", "sample"))  # sample | postgres
    # Tiền tố dựng URL bài viết từ slug (rỗng -> dùng "/{slug}")
    article_base_url: str = field(default_factory=lambda: _env("ARTICLE_BASE_URL", ""))
    ingest_batch_size: int = field(default_factory=lambda: _env_int("INGEST_BATCH_SIZE", 500))

    # --- Postgres CMS (SOURCE=postgres) ---
    pg_host: str = field(default_factory=lambda: _env("PG_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: _env_int("PG_PORT", 5432))
    pg_user: str = field(default_factory=lambda: _env("PG_USER", ""))
    # Chấp nhận cả PG_PASS (theo .env dự án) lẫn PG_PASSWORD
    pg_password: str = field(default_factory=lambda: _env("PG_PASS", "") or _env("PG_PASSWORD", ""))
    pg_db: str = field(default_factory=lambda: _env("PG_DB", ""))
    pg_schema: str = field(default_factory=lambda: _env("PG_SCHEMA", "public"))
    pg_connect_timeout: int = field(default_factory=lambda: _env_int("PG_CONNECT_TIMEOUT", 10))

    # ============================================================================
    # TÍNH NĂNG PHỤ — bật/tắt có trách nhiệm: MẶC ĐỊNH TẮT/AN TOÀN, thiếu phụ
    # thuộc (Redis/OpenSearch/OpenAI) thì degrade gracefully thay vì crash.
    # ============================================================================

    # --- Lexical backend production (F-04) ---
    # local (BM25 thuần Python) | opensearch — đã khai báo ở trên (lexical_backend)
    opensearch_url: str = field(default_factory=lambda: _env("OPENSEARCH_URL", "http://localhost:9200"))
    opensearch_index: str = field(default_factory=lambda: _env("OPENSEARCH_INDEX", "news_articles"))
    opensearch_user: str = field(default_factory=lambda: _env("OPENSEARCH_USER", ""))
    opensearch_password: str = field(default_factory=lambda: _env("OPENSEARCH_PASSWORD", ""))

    # --- Redis (dùng chung cho cache & feedback backend) ---
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))

    # --- (GĐ1) Log click/dwell — nền tảng đánh giá & học xếp hạng ---
    feedback_enabled: bool = field(default_factory=lambda: _env_bool("FEEDBACK_ENABLED", False))
    feedback_backend: str = field(default_factory=lambda: _env("FEEDBACK_BACKEND", "jsonl"))  # jsonl | redis
    feedback_path: str = field(
        default_factory=lambda: _env("FEEDBACK_PATH", os.path.join(BASE_DIR, "data", "feedback", "events.jsonl"))
    )

    # --- (GĐ3) Query understanding ---
    qu_spellcorrect: bool = field(default_factory=lambda: _env_bool("QU_SPELLCORRECT", False))
    qu_expansion: bool = field(default_factory=lambda: _env_bool("QU_EXPANSION", False))
    qu_llm_rewrite: bool = field(default_factory=lambda: _env_bool("QU_LLM_REWRITE", False))
    qu_max_edit_distance: int = field(default_factory=lambda: _env_int("QU_MAX_EDIT_DISTANCE", 2))
    synonyms_path: str = field(default_factory=lambda: _env("SYNONYMS_PATH", ""))  # file JSON tùy chọn

    # --- (GĐ5) Cache truy vấn nóng ---
    cache_enabled: bool = field(default_factory=lambda: _env_bool("CACHE_ENABLED", False))
    cache_backend: str = field(default_factory=lambda: _env("CACHE_BACKEND", "memory"))  # memory | redis
    cache_ttl_seconds: int = field(default_factory=lambda: _env_int("CACHE_TTL_SECONDS", 60))
    cache_max_size: int = field(default_factory=lambda: _env_int("CACHE_MAX_SIZE", 1000))
    cache_bypass_fresh: bool = field(default_factory=lambda: _env_bool("CACHE_BYPASS_FRESH", True))

    # --- (GĐ5) Metrics / dashboard / alert ---
    metrics_enabled: bool = field(default_factory=lambda: _env_bool("METRICS_ENABLED", True))
    dashboard_enabled: bool = field(default_factory=lambda: _env_bool("DASHBOARD_ENABLED", True))
    alert_p95_ms: float = field(default_factory=lambda: _env_float("ALERT_P95_MS", 500.0))
    alert_error_rate: float = field(default_factory=lambda: _env_float("ALERT_ERROR_RATE", 0.05))
    alert_zero_result_rate: float = field(default_factory=lambda: _env_float("ALERT_ZERO_RESULT_RATE", 0.3))
    alert_webhook_url: str = field(default_factory=lambda: _env("ALERT_WEBHOOK_URL", ""))

    # --- (GĐ5) A/B & interleaving ---
    experiment_enabled: bool = field(default_factory=lambda: _env_bool("EXPERIMENT_ENABLED", False))

    # --- Feature Toggles (Bật/tắt chức năng phụ) ---
    dedup_enabled: bool = field(default_factory=lambda: _env_bool("DEDUP_ENABLED", True))
    time_decay_enabled: bool = field(default_factory=lambda: _env_bool("TIME_DECAY_ENABLED", True))
    mmr_enabled: bool = field(default_factory=lambda: _env_bool("MMR_ENABLED", True))

    # --- Admin (blue-green reindex) ---
    admin_token: str = field(default_factory=lambda: _env("ADMIN_TOKEN", ""))  # rỗng -> route admin tắt

    # --- Demo UI ---
    ui_enabled: bool = field(default_factory=lambda: _env_bool("UI_ENABLED", True))
    # Tự nạp bài mẫu lúc khởi động nếu chỉ mục rỗng & SOURCE=sample (tiện demo Docker)
    autoload_sample: bool = field(default_factory=lambda: _env_bool("AUTOLOAD_SAMPLE", False))

    def pg_conninfo(self) -> dict:
        """Tham số kết nối psycopg (dict, tránh escape ký tự đặc biệt trong mật khẩu)."""
        return {
            "host": self.pg_host,
            "port": self.pg_port,
            "user": self.pg_user,
            "password": self.pg_password,
            "dbname": self.pg_db,
            "connect_timeout": self.pg_connect_timeout,
        }

    @classmethod
    def from_env(cls) -> "Settings":
        return cls()
