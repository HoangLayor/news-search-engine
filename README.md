# News Search Engine

Hệ thống tìm kiếm cho trang báo chí số — hiện thực hóa theo
[docs/BaoCao-ThietKe.md](docs/BaoCao-ThietKe.md) và [docs/KeHoach-ThucHien.md](docs/KeHoach-ThucHien.md).

Ánh xạ truy vấn (từ khóa hoặc câu hỏi tiếng Việt) → danh sách bài viết xếp hạng
theo mức độ liên quan, qua kiến trúc **phễu nhiều tầng**: hiểu truy vấn → truy hồi
lexical (BM25) ⨁ semantic (vector) → hợp nhất RRF → ưu tiên độ mới (QDF) → rerank
tinh → gom cụm & đa dạng hóa → JSON 5 trường.

Mọi thành phần có **backend local thuần Python (chỉ stdlib + numpy)** để chạy
offline cho dev/test/demo; production đổi sang OpenSearch/Milvus/OpenAI/BGE qua
biến môi trường mà **không đổi code** (import guard + factory).

---

## Kiến trúc

```
                       ┌──────────────────────────┐
  CMS / Nguồn bài ───▶ │  Ingest & Chuẩn hóa       │  F-02, F-03
  (dict / JSON)        │  clean_html · tokenizer   │
                       └────────────┬──────────────┘
                                    │  Article
                 ┌──────────────────┼──────────────────┬───────────────┐
                 ▼                  ▼                   ▼               ▼
          ┌────────────┐    ┌──────────────┐    ┌────────────┐   ┌──────────┐
          │ LexicalIndex│    │ Embedder +   │    │ MinHash    │   │Knowledge │
          │ BM25  F-04  │    │ VectorIndex  │    │ Deduper    │   │Graph     │
          │             │    │ F-05         │    │ F-07       │   │F-06      │
          └─────┬───────┘    └──────┬───────┘    └────────────┘   └──────────┘
                │                   │              (IndexManager: F-01, F-08)
   Query ─▶ parse_query ─▶ retrieval(lexical|semantic|hybrid) ─▶ RRF (F-11)
            F-09..F-12                    │
                                          ▼
                        apply_time_decay (QDF, F-13)
                                          ▼
                        rerank (Noop | BGE)         [GĐ 3]
                                          ▼
                        collapse_clusters + MMR (F-14)
                                          ▼
                        make_snippet (F-15)
                                          ▼
                     list[SearchResultItem]  →  JSON 5 trường
              (article_id, title, url, published_at, snippet)
```

## Ánh xạ chức năng F-01…F-15 → mã nguồn

| Mã  | Chức năng                          | File                                                                                                 |
| ---- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| F-01 | Khởi tạo cơ sở dữ liệu         | [manager.py](news_search/index/manager.py) `IndexManager.__init__`                                  |
| F-02 | Thu nhận & chuẩn hóa bài         | [ingest/cleaner.py](news_search/ingest/cleaner.py)                                                    |
| F-03 | Tách & token hóa tiếng Việt      | [ingest/tokenizer.py](news_search/ingest/tokenizer.py)                                                |
| F-04 | Đánh chỉ mục BM25                | [index/lexical.py](news_search/index/lexical.py)                                                      |
| F-05 | Sinh embedding + vector index        | [index/embeddings.py](news_search/index/embeddings.py), [index/vector.py](news_search/index/vector.py) |
| F-06 | Trích xuất thực thể + KG         | [index/entities.py](news_search/index/entities.py)                                                    |
| F-07 | Phát hiện trùng/gần trùng       | [index/dedup.py](news_search/index/dedup.py)                                                          |
| F-08 | Cập nhật / gỡ bài đồng bộ     | [manager.py](news_search/index/manager.py) `remove_article`                                         |
| F-09 | Tìm bằng từ khóa                 | [index/lexical.py](news_search/index/lexical.py) `search`                                           |
| F-10 | Tìm bằng câu hỏi tự nhiên      | [index/vector.py](news_search/index/vector.py) `search`                                             |
| F-11 | Hybrid + RRF                         | [search/fusion.py](news_search/search/fusion.py), [search/pipeline.py](news_search/search/pipeline.py) |
| F-12 | Lọc theo metadata                   | [models.py](news_search/models.py) `SearchFilters`, `ArticleStore.filter_ids`                     |
| F-13 | Ưu tiên độ mới (time-decay/QDF) | [search/ranking.py](news_search/search/ranking.py)                                                    |
| F-14 | Gom cụm & đa dạng hóa (MMR)      | [search/diversify.py](news_search/search/diversify.py)                                                |
| F-15 | Trích đoạn nổi bật (snippet)    | [search/snippet.py](news_search/search/snippet.py)                                                    |

Điều phối: [search/pipeline.py](news_search/search/pipeline.py) · HTTP API:
[api/app.py](news_search/api/app.py) · Hợp đồng interface:
[CONTRACTS.md](CONTRACTS.md).

---

## Cấu trúc thư mục

```
news-search-engine/
├── pyproject.toml            # packaging + nhóm phụ thuộc tùy chọn ([postgres],[bge],[milvus]...)
├── .env / .env.example       # cấu hình runtime (config.py tự nạp .env)
├── data/sample_articles.json # dữ liệu mẫu offline
├── docs/                     # BaoCao-ThietKe, KeHoach-ThucHien, CONTRACTS
├── notebooks/                # demo_chi_tiet.ipynb (kiểm thử từng chức năng)
├── scripts/                  # run_demo · benchmark · index_from_postgres
├── tests/                    # 116 test (pytest)
└── news_search/
    ├── config.py  models.py
    ├── sources/              # ★ tầng nguồn dữ liệu: base · postgres · sample (+ factory)
    ├── ingest/               # F-02/03: làm sạch HTML + tokenize tiếng Việt
    ├── index/                # F-04/05/06/07: BM25 · embeddings · vector/milvus · dedup · entities · manager
    ├── search/               # F-11/13/14/15: query · fusion · ranking · rerank · diversify · snippet · pipeline
    └── api/                  # FastAPI
```

Thêm nguồn/backend mới = thêm 1 file + 1 nhánh factory (`get_source`, `get_embedder`,
`get_vector_index`), không đụng tầng khác.

## Cài đặt & chạy

```bash
# 1. Tạo venv + cài (core; extras theo backend cần dùng)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[postgres,bge,milvus,dev]"
#   hoặc tối thiểu:  pip install -r requirements.txt

# 2. Demo 5 truy vấn ví dụ (offline, hash/local)
.venv\Scripts\python.exe scripts\run_demo.py

# 3. Toàn bộ test (hermetic: conftest ép local/hash)
.venv\Scripts\python.exe -m pytest tests -q

# 4. Nạp bài từ Postgres CMS rồi tìm thử
.venv\Scripts\python.exe scripts\index_from_postgres.py --source postgres --limit 200 --query "kinh tế"

# 5. Benchmark lexical/dense/hybrid
.venv\Scripts\python.exe scripts\benchmark.py

# 6. Khởi động API (app dựng lazy theo .env)
.venv\Scripts\python.exe -m uvicorn news_search.api.app:app --port 8000
```

## Docker & Demo UI

Chạy demo tương tác **offline, tức thì** (image nhẹ — chỉ core deps, KHÔNG torch/model):

```bash
docker compose up --build          # -> mở http://localhost:8000
# Tùy chọn có Redis (cache/feedback dùng chung):
CACHE_BACKEND=redis FEEDBACK_BACKEND=redis docker compose --profile full up --build
# Tùy chọn có Milvus + Attu (quan sát/quản lý vector):
docker compose -f docker-compose.yml -f docker-compose.milvus.yml up --build
```

Container tự chạy backend `local/hash`, **tự nạp bài mẫu** (`AUTOLOAD_SAMPLE`), bật
feedback + cache + metrics + UI. Không cần tải model hay dịch vụ ngoài. Nút *Reindex*
trong UI cần bật admin: `ADMIN_TOKEN=<bí-mật> docker compose up` (mặc định tắt an toàn).

**Milvus + Attu** ([docker-compose.milvus.yml](docker-compose.milvus.yml)): dựng
Milvus standalone (etcd + minio + milvus) và **Attu** — web UI chính thức để quan
sát/quản lý collection, index, vector. App chuyển `VECTOR_BACKEND=milvus` và nạp
vector mẫu vào collection `news_articles`. Sau khi lên: **Attu tại http://localhost:8001**
(kết nối sẵn `milvus-standalone:19530`), Milvus gRPC `localhost:19530`.

**Demo UI** ([news_search/api/ui.html](news_search/api/ui.html)) tại `GET /` (hoặc `/ui`):
tìm kiếm (chọn mode + lọc chuyên mục/tác giả/thời gian), kết quả có highlight,
**click ghi log** (feedback → CTR), thêm bài viết, reindex, **chỉ số trực tiếp** tự
làm mới, và **🔬 chế độ Giải thích** — bật để xem *từng bước pipeline* (hiểu truy vấn
→ BM25 → vector → RRF → time-decay → rerank → gom cụm → MMR → kết quả) kèm **danh
sách trung gian + điểm số** ở mỗi tầng (endpoint `GET /explain`). Render an toàn XSS.
Tắt UI bằng `UI_ENABLED=false`.

Chạy UI không cần Docker: `uvicorn news_search.api.app:app --port 8000` (đặt
`AUTOLOAD_SAMPLE=true SOURCE=sample` để có sẵn dữ liệu) rồi mở `http://localhost:8000`.

### API

| Method & path                                                                 | Mô tả                                                                                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `POST /articles`                                                            | Nạp 1 bài (dict). Thiếu field bắt buộc → 422                                                                        |
| `POST /articles/bulk`                                                       | Nạp một lô bài                                                                                                        |
| `DELETE /articles/{id}`                                                     | Gỡ bài khỏi mọi chỉ mục (F-08)                                                                                      |
| `GET /articles/{id}/entities`                                               | Thực thể của bài (F-06)                                                                                               |
| `GET /search?q=&mode=&top_k=&author=&category=&source=&date_from=&date_to=` | Trả JSON array, mỗi phần tử ĐÚNG 5 trường.`q` rỗng → 400. Header `X-Search-Id` khi bật feedback            |
| `GET /explain?q=&mode=&...`                                                 | **Giải thích pipeline**: `{query, mode, results, trace}` — từng bước + kết quả trung gian (bỏ qua cache) |
| `POST /events/click`                                                        | Log click/dwell (GĐ1) —`{search_id, article_id, position, dwell_ms?}`                                                 |
| `GET /stats`                                                                | JSON tổng hợp cho UI: counts + backends + features + metrics + CTR                                                      |
| `GET /metrics` · `GET /dashboard`                                        | Prometheus text · dashboard HTML (GĐ5)                                                                                  |
| `POST /admin/reindex`                                                       | Reindex blue-green (header`X-Admin-Token`)                                                                              |
| `GET /` · `GET /ui`                                                      | **Demo UI** tương tác (`UI_ENABLED`)                                                                           |
| `GET /healthz`                                                              | Sức khỏe + số lượng đã index                                                                                       |

Ví dụ:

```bash
curl "http://localhost:8000/search?q=lam%20phat&top_k=5"        # không dấu vẫn ra "lạm phát"
curl "http://localhost:8000/search?q=động%20đất"                # tin nóng lên đầu
curl "http://localhost:8000/search?q=giá%20xăng&category=Kinh%20tế"
```

---

## Cấu hình & production backend

Chọn backend qua biến môi trường (xem [.env.example](.env.example)). Mặc định là
`local`/`hash`/`none` (offline). Chuyển production:

| Biến               | Local / Dev                       | Production (khuyến nghị)                                                                                               |
| ------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `LEXICAL_BACKEND` | `local` (BM25 thuần Python)    | `opensearch`                                                                                                           |
| `VECTOR_BACKEND`  | `local` (brute-force cosine)    | `milvus` (chỉ mục **HNSW**)                                                                                    |
| `EMBEDDER`        | `hash` (offline, deterministic) | **`vi`** (`AITeamVN/Vietnamese_Embedding` — VN chuyên biệt, **mặc định**) · `bge` · `openai` |
| `RERANKER`        | `none`                          | **`vi`** (`AITeamVN/Vietnamese_Reranker`) · `bge`                                                           |
| `TOKENIZER`       | `regex` (test)                  | **`auto`** → `pyvi` (tách từ ghép "bất_động_sản")                                                      |
| `DEDUP_BACKEND`   | `local` (thuần Python)         | `datasketch` (nhanh hơn ở quy mô)                                                                                   |

> **Nâng cấp tiếng Việt chuyên biệt** ([embeddings.py](news_search/index/embeddings.py) `VietnameseEmbedder`,
> [rerank.py](news_search/search/rerank.py) `VietnameseReranker`): mô hình fine-tune từ
> BGE-m3 cho tiếng Việt → chất lượng semantic + top-k cao hơn bge gốc, **cùng hạ tầng**.
> `pip install -e ".[vi,dedup]"` rồi đặt `EMBEDDER=vi RERANKER=vi TOKENIZER=auto DEDUP_BACKEND=datasketch`.
> Model tải lần đầu (~2.6GB); bật GPU `BGE_DEVICE=cuda` cho index lô lớn.

Cài thêm khi cần: bỏ comment các dòng tương ứng trong
[requirements.txt](requirements.txt) (`opensearch-py`, `pymilvus`, `openai`,
`FlagEmbedding`/`sentence-transformers`, `underthesea`, `neo4j`). Knowledge graph
in-memory ([entities.py](news_search/index/entities.py)) thay bằng Neo4j ở
production.

**BGE-m3 local** ([embeddings.py](news_search/index/embeddings.py) `BGEEmbedder`):
`EMBEDDER=bge` — dense 1024 chiều đa ngữ, ưu tiên `FlagEmbedding.BGEM3FlagModel`,
fallback `sentence-transformers`. Bật GPU qua `BGE_DEVICE=cuda` cho index lô lớn.

**Milvus + HNSW** ([milvus_vector.py](news_search/index/milvus_vector.py)
`MilvusVectorIndex`): `VECTOR_BACKEND=milvus`. Chỉ mục `HNSW` với `M`,
`efConstruction`, `ef` cấu hình qua `HNSW_*` (xem [.env.example](.env.example)).
Cần Milvus server (`docker run -p 19530:19530 milvusdb/milvus:latest`) hoặc
Milvus Lite (`MILVUS_URI=./milvus.db`). *Chưa có test offline — cần server thật.*

## Nguồn dữ liệu (data source)

Tầng [`news_search/sources/`](news_search/sources/) trừu tượng hóa nơi bài viết
đến từ đâu — mọi nguồn trả về **dict thô** tương thích `normalize_article`, nên
index/search không phụ thuộc nguồn. Chọn qua `SOURCE`:

| `SOURCE`   | Lớp                  | Mô tả                                                   |
| ------------ | --------------------- | --------------------------------------------------------- |
| `sample`   | `SampleJsonSource`  | Đọc`data/sample_articles.json` (offline)              |
| `postgres` | `PostgresCMSSource` | CMS thật:`public.articles` + join category/author/tags |

`PostgresCMSSource` ánh xạ schema CMS (`articles.content` HTML → body, `publish_date`
→ published_at, category `is_major`, tác giả gộp từ `article_authors`), chỉ lấy bài
`deleted_at IS NULL AND publish_date IS NOT NULL`, **keyset pagination theo id**
(thân thiện pgbouncer cổng 6432), và `deleted_ids_since()` để đồng bộ xóa (F-08).
Cấu hình `PG_HOST/PG_PORT/PG_USER/PG_PASS/PG_DB/PG_SCHEMA` trong `.env`; cài driver
qua `pip install -e ".[postgres]"`.

> **Lưu ý môi trường:** khi dùng `EMBEDDER=bge` (torch) cùng `SOURCE=postgres`
> (psycopg) trên Windows có thể xung đột OpenMP → segfault. `scripts/index_from_postgres.py`
> đã xử lý bằng cách nạp torch trước psycopg. Nếu vẫn gặp, đặt `KMP_DUPLICATE_LIB_OK=TRUE`.

### Benchmark lexical vs dense vs hybrid

```bash
.venv\Scripts\python.exe scripts\benchmark.py                 # embedder theo .env
# EMBEDDER=hash .venv\Scripts\python.exe scripts\benchmark.py   # ép hash cho nhanh
```

Đo **hiệu quả** ở tầng truy hồi (Recall@10, MRR, nDCG@10 trên bộ qrels 10 truy
vấn) và **tốc độ** pipeline end-to-end (p50/p95 ms). Ví dụ với `hash` (corpus 29
bài): lexical ~0.985 nDCG / 3.5ms, dense ~0.953 / 6.4ms, hybrid ~0.979 / 6.5ms —
lexical vượt trội trên corpus nhỏ sạch; dense sẽ mạnh hơn rõ khi dùng BGE-m3 thật.

## Tính năng phụ (bật/tắt có trách nhiệm)

Các năng lực nâng cao đều là **feature flag**: mặc định TẮT/an toàn, không đổi
hành vi lõi, và **degrade gracefully** khi thiếu phụ thuộc (Redis/OpenSearch/
OpenAI) thay vì crash. Bật qua `.env` (xem [.env.example](.env.example)).

| Cờ                            | Mặc định  | Tính năng                                                                                                                                            | Module                                                          |
| ------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `FEEDBACK_ENABLED`           | off          | **Log click/dwell** (GĐ1): `/search` trả `X-Search-Id`, `POST /events/click` ghi log; backend jsonl/redis; CTR/zero-result cho dashboard | [feedback/](news_search/feedback/)                               |
| `QU_SPELLCORRECT`            | off          | **Sửa lỗi chính tả** (GĐ3): Norvig trên vocab corpus, chỉ sửa từ OOV                                                                    | [understanding.py](news_search/search/understanding.py)          |
| `QU_EXPANSION`               | off          | **Mở rộng đồng nghĩa/alias** (GĐ3)                                                                                                         | ↑                                                              |
| `QU_LLM_REWRITE`             | off          | **Viết lại truy vấn bằng LLM** (cần OpenAI key; thiếu -> bỏ qua)                                                                          | ↑                                                              |
| `CACHE_ENABLED`              | off          | **Cache truy vấn nóng** (GĐ5): TTL+LRU memory / Redis; vô hiệu theo generation; bỏ qua tin nóng                                           | [cache.py](news_search/service/cache.py)                         |
| `METRICS_ENABLED`            | **on** | **Metrics** (GĐ5): `/metrics` (Prometheus), p50/p95, error/zero-result/cache-hit rate                                                         | [metrics.py](news_search/service/metrics.py)                     |
| `DASHBOARD_ENABLED`          | on           | **Dashboard HTML** `/dashboard` + cảnh báo ngưỡng (p95/error/zero-result)                                                                  | ↑                                                              |
| `EXPERIMENT_ENABLED`         | off          | **A/B**: gán biến thể (`X-Variant`) + log để phân tích online                                                                           | [app.py](news_search/api/app.py)                                 |
| `ADMIN_TOKEN`                | (rỗng=tắt) | **Blue-green reindex** `POST /admin/reindex` (dựng chỉ mục mới rồi hoán đổi nguyên tử)                                               | [search_service.py](news_search/service/search_service.py)       |
| `LEXICAL_BACKEND=opensearch` | local        | **OpenSearch** BM25 production                                                                                                                   | [opensearch_lexical.py](news_search/index/opensearch_lexical.py) |

Ngoài ra: `IndexManager.snapshot(path)` / `restore(path)` cho **bền vững** (lưu bài
gốc ra JSONL, dựng lại mọi chỉ mục khi phục hồi) — chống mất dữ liệu cho backend
in-memory.

Ví dụ bật vài tính năng:

```bash
FEEDBACK_ENABLED=true CACHE_ENABLED=true QU_SPELLCORRECT=true \
  .venv\Scripts\python.exe -m uvicorn news_search.api.app:app --port 8000
# rồi: GET /dashboard · GET /metrics · POST /events/click
```

## Lưu ý về chất lượng ngữ nghĩa (local)

`HashingEmbedder` (fallback offline, char n-gram) chỉ là **xấp xỉ ngữ nghĩa** —
với truy vấn rất ngắn (1–2 từ) độ phân giải thấp và có thể lẫn nhiễu vào nhánh
hybrid. Đây là hạn chế của backend offline dùng cho demo; production dùng
`text-embedding-3-large` cho chất lượng ngữ nghĩa thực sự. Tầng time-decay được
thiết kế theo QDF: truy vấn thường decay nhẹ (không vùi bài liên quan cũ), truy
vấn tin nóng (`hôm nay`, `mới nhất`…) mới cho recency áp đảo.

## Kiểm thử

`pytest tests` — **146 test** phủ: ingest, BM25 + snippet, embedding + vector,
query understanding (+ spell-correct/expansion) + RRF + time-decay + rerank,
dedup + MMR, NER + KG, pipeline e2e (checklist §7.3), HTTP API, nguồn dữ liệu
(sample + mapping Postgres), và tính năng phụ (feedback/CTR, cache, metrics/alert,
snapshot/reindex, factory backend). Toàn bộ deterministic & hermetic (conftest ép
local/hash; `now` cố định; không phụ thuộc mạng/giờ thực).
