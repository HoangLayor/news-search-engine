# Kế hoạch thực hiện — News Search Engine

> Kế hoạch triển khai theo giai đoạn (phased rollout) cho hệ thống tìm kiếm báo chí số. Nguyên tắc: **giao giá trị sớm với chi phí thấp**, thêm tầng khi đo được điểm đau thực tế. Tham chiếu tài liệu thiết kế: [BaoCao-ThietKe.md](BaoCao-ThietKe.md).

## Mục lục
1. [Nguyên tắc & mục tiêu](#1-nguyên-tắc--mục-tiêu)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Lộ trình theo giai đoạn](#3-lộ-trình-theo-giai-đoạn)
4. [Chi tiết công việc theo module](#4-chi-tiết-công-việc-theo-module)
5. [Mốc bàn giao & tiêu chí ra quyết định](#5-mốc-bàn-giao--tiêu-chí-ra-quyết-định)
6. [Rủi ro & giảm thiểu](#6-rủi-ro--giảm-thiểu)
7. [Nhân sự & ước tính thời gian](#7-nhân-sự--ước-tính-thời-gian)

---

## 1. Nguyên tắc & mục tiêu

- **Bắt đầu bằng thứ giải quyết 80% giá trị với 20% công sức**: BM25 + metadata + time-decay đã bao phủ phần lớn nhu cầu.
- **Không nhảy thẳng vào vector/LLM**: chỉ thêm khi đo được BM25 không đủ.
- **Đo lường từ ngày đầu**: log click/vị trí/dwell time là tài sản không tạo lại được.
- **Mỗi giai đoạn có mốc nghiệm thu rõ ràng** (tham chiếu checklist trong tài liệu thiết kế).

**Mục tiêu chất lượng (chốt lại với khách hàng trước khi bắt đầu):**
- Latency tìm kiếm: p95 < 300ms, p99 < 500ms
- Độ trễ index bài mới: ≤ 2 phút (chốt SLA)
- nDCG@10 ≥ 0.6 trên bộ đánh giá (baseline điều chỉnh sau)

---

## 2. Kiến trúc tổng thể

```
                         ┌─────────────────────────┐
   CMS / Nguồn bài  ───▶ │  Ingest & Chuẩn hóa      │  F-02, F-03
   (webhook/queue)       │  (làm sạch, token hóa)   │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
       ┌────────────┐          ┌────────────┐          ┌────────────┐
       │ BM25 index │          │ Embedding  │          │ NER + dedup│
       │ OpenSearch │          │ → Milvus   │          │ Neo4j/LSH  │
       │  F-04      │          │  F-05      │          │ F-06,F-07  │
       └─────┬──────┘          └─────┬──────┘          └────────────┘
             │                       │
   Query ─▶ [Query Understanding] ─▶ [Retrieval: BM25 ⨁ Vector] ─▶ [RRF fusion]
             F-09..F-12                       │
                                              ▼
                              [Ranking: time-decay + tín hiệu click]  F-13
                                              ▼
                              [Rerank: BGE-reranker]  F-11
                                              ▼
                              [Post: dedup + MMR + snippet]  F-14, F-15
                                              ▼
                                   JSON (5 trường thống nhất)
```

---

## 3. Lộ trình theo giai đoạn

| GĐ | Tên | Mục tiêu chính | Thời lượng ước tính | Kết quả bàn giao |
|---|---|---|---|---|
| **0** | Khởi động | Chốt yêu cầu, dựng hạ tầng, ingest thử | ~1 tuần | Môi trường dev + pipeline ingest tối thiểu |
| **1** | MVP Lexical | BM25 + metadata + time-decay + snippet | ~2 tuần | Search box chạy được, giá trị thực tế đầu tiên |
| **2** | Hybrid | Embedding + Milvus + RRF fusion | ~2 tuần | Tìm câu hỏi tự nhiên, hybrid search |
| **3** | Rerank & chất lượng | BGE-reranker + query rewrite + đánh giá | ~1.5 tuần | Chất lượng top-10 tăng rõ, có bộ đánh giá |
| **4** | Đặc thù báo chí | Dedup/gom cụm + MMR + NER + knowledge graph | ~2 tuần | Xử lý trùng lặp, đa dạng hóa, lọc thực thể |
| **5** | Vận hành & tối ưu | Monitoring, cache, A/B, tối ưu chi phí | ~1.5 tuần | Dashboard, cảnh báo, quy trình cải tiến liên tục |

> Tổng ước tính: **~10 tuần** cho một đội nhỏ. Có thể dừng/bàn giao ở cuối bất kỳ giai đoạn nào vì mỗi GĐ đều tạo ra hệ thống chạy được.

---

## 4. Chi tiết công việc theo module

### Giai đoạn 0 — Khởi động
- [ ] Chốt quy mô thực tế (số bài hiện có, tốc độ tăng, QPS, SLA)
- [ ] Chốt schema metadata bắt buộc: `title, body, author, category, published_at, url, status`
- [ ] Dựng hạ tầng dev: OpenSearch (có replica + snapshot), Postgres/Milvus, Redis (cache + queue)
- [ ] Kết nối nguồn bài (webhook/queue từ CMS) — xử lý cả sự kiện **publish / update / unpublish**
- [ ] Pipeline ingest tối thiểu: làm sạch HTML, chuẩn hóa metadata (F-02)

### Giai đoạn 1 — MVP Lexical (giá trị sớm nhất)
- [ ] Token hóa tiếng Việt (analyzer OpenSearch + underthesea nếu cần) (F-03)
- [ ] Mapping & inverted index BM25 (F-04)
- [ ] API search: từ khóa + `function_score` time-decay (Gauss/exp theo `published_at`) (F-09, F-13)
- [ ] Lọc metadata: tác giả / chuyên mục / khoảng thời gian / nguồn (F-12)
- [ ] Sinh snippet/highlight (F-15)
- [ ] Đồng bộ gỡ bài: bài `unpublish` biến mất khỏi kết quả ngay (F-08)
- [ ] **Log click/vị trí/dwell time từ đây** (chuẩn bị cho đánh giá)
- [ ] Định dạng response JSON 5 trường thống nhất

### Giai đoạn 2 — Hybrid Retrieval
- [ ] Sinh embedding bài viết (text-embedding-3-large hoặc BGE-m3 local) (F-05)
- [ ] Lưu vector vào Milvus, dựng ANN index (HNSW) — cân nhắc quantize
- [ ] API tìm câu hỏi tự nhiên (semantic) (F-10)
- [ ] Kết hợp lexical + dense bằng **RRF** (F-11)
- [ ] Backfill embedding cho kho bài hiện có (job batch)

### Giai đoạn 3 — Rerank & Đánh giá
- [ ] Tích hợp BGE-reranker-v2-m3 rerank top ~50–100 (self-host)
- [ ] Query understanding: sửa lỗi chính tả + viết lại/mở rộng truy vấn (gpt-4.1-mini) — chỉ áp dụng có chọn lọc để tiết kiệm
- [ ] Xây **bộ đánh giá**: tập query mẫu + nhãn (thủ công + LLM-as-judge)
- [ ] Đo nDCG@10 / MRR, thiết lập baseline
- [ ] Regression set để kiểm thử mỗi lần đổi ranking

### Giai đoạn 4 — Đặc thù báo chí
- [ ] Phát hiện trùng/gần trùng: MinHash-LSH / SimHash (F-07)
- [ ] Gom cụm sự kiện + đa dạng hóa kết quả bằng MMR (F-14)
- [ ] NER trích xuất thực thể (gpt-4.1-mini hoặc underthesea) (F-06)
- [ ] Knowledge graph Neo4j: liên kết người/tổ chức/sự kiện/bài — hỗ trợ lọc theo thực thể
- [ ] Phân loại intent truy vấn (thời sự / thông tin / điều hướng) tinh chỉnh time-decay

### Giai đoạn 5 — Vận hành & Tối ưu
- [ ] Dashboard: latency, error rate, index lag, CTR, zero-result rate
- [ ] Cảnh báo (alert) khi vượt ngưỡng
- [ ] Cache kết quả truy vấn phổ biến (tin nóng) — giảm chi phí rerank/LLM
- [ ] Cơ chế A/B hoặc interleaving để so phiên bản ranking
- [ ] Quy trình reindex toàn bộ (blue-green index khi đổi model embedding)
- [ ] Review query "0 kết quả" định kỳ → bổ sung từ đồng nghĩa/alias

---

## 5. Mốc bàn giao & tiêu chí ra quyết định

| Mốc | Điều kiện hoàn thành | Quyết định tiếp theo |
|---|---|---|
| Cuối GĐ 1 | Checklist 7.1 + 7.2 (phần lexical) + test keyword/filter/freshness đạt | Nếu BM25 đủ tốt cho phần lớn query → cân nhắc hoãn GĐ 2 |
| Cuối GĐ 2 | Test natural-query + hybrid đạt; latency p95 < 300ms | Đo tỷ lệ query cần semantic để quyết mức đầu tư rerank |
| Cuối GĐ 3 | nDCG@10 ≥ baseline; regression set xanh | Chốt có cần NER/knowledge graph ngay không |
| Cuối GĐ 4 | Test dedup/đa dạng đạt; không ≥ 2 bài trùng trong top-10 | — |
| Cuối GĐ 5 | Toàn bộ checklist 7.4 đạt; dashboard + alert hoạt động | Go-live production |

---

## 6. Rủi ro & giảm thiểu

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Không có ground-truth chuẩn | Khó đo chất lượng | Dùng LLM-as-judge + log click + interleaving; nghiệm thu theo baseline tương đối, cam kết vòng lặp cải tiến |
| Chi phí API tăng theo QPS | Ngân sách khó kiểm soát | Cache query nóng; rerank local; NER local (underthesea); chỉ LLM rewrite có chọn lọc |
| Tokenization tiếng Việt kém | Sai kết quả tên riêng | Kiểm thử riêng cho tiếng Việt; kết hợp analyzer + underthesea; bộ alias |
| Đổi model embedding cần reindex | Downtime | Blue-green index; lên lịch reindex ngoài giờ cao điểm |
| Bài gỡ vẫn hiển thị | Rủi ro pháp lý/biên tập | Ưu tiên F-08 ngay GĐ 1; test case gỡ bài bắt buộc |
| Trôi dữ liệu (drift) theo thời sự | Giảm chất lượng dần | Định kỳ đánh giá lại; review zero-result; cập nhật từ điển đồng nghĩa |

---

## 7. Nhân sự & ước tính thời gian

**Đội tối thiểu:**
- 1 Backend/Search engineer (chủ lực, xuyên suốt)
- 1 ML/NLP engineer (GĐ 2–4: embedding, rerank, NER, đánh giá)
- 0.5 DevOps (hạ tầng, monitoring — GĐ 0 và GĐ 5)
- 0.5 QA/BTV (nghiệm thu, xây bộ query đánh giá)

**Tổng thời gian:** ~10 tuần (có thể rút ngắn nếu dừng ở GĐ 1–2 hoặc dùng managed service thay self-host).

**Đòn bẩy tiết kiệm cho startup:**
- Dùng managed OpenSearch/Zilliz Cloud thay tự vận hành cụm ở giai đoạn đầu
- Bỏ Neo4j/knowledge graph nếu chưa cần lọc theo thực thể (thêm ở GĐ 4 khi có nhu cầu)
- NER local (underthesea) thay GPT để đưa chi phí index về gần $0.0002/bài

---

*Bước tiếp theo đề xuất: chốt **quy mô thực tế + SLA** ở Giai đoạn 0, sau đó bắt tay dựng MVP Lexical (GĐ 1) — phần tạo giá trị nhanh nhất với chi phí thấp nhất.*

---

## 8. Trạng thái hiện thực hóa (cập nhật 2026-07-08)

Đã dựng codebase chạy được (`news_search/`, 111 test pass, demo + benchmark +
API thật). Backend mặc định là **local thuần Python** (offline); production đổi
qua env. Dưới đây là đối chiếu với kế hoạch — ✅ xong · 🟡 một phần · ❌ thiếu.

| GĐ | Hạng mục | TT | Ghi chú |
|----|----------|----|---------|
| 0 | Schema metadata bắt buộc | ✅ | `models.REQUIRED_FIELDS`, `cleaner.normalize_article` |
| 0 | Hạ tầng OpenSearch/Milvus/Redis | 🟡 | Milvus+HNSW đã có code (chưa test server thật); **OpenSearch chưa hiện thực**; **Redis chưa có** |
| 0 | Kết nối CMS (webhook publish/update/unpublish) | 🟡 | Logic index/update/remove (F-08) ✅; **chưa có connector webhook/queue thật** (mới có API dict/JSON + bulk) |
| 0 | Pipeline ingest làm sạch | ✅ | `cleaner.clean_html` |
| 1 | BM25 + lọc metadata + snippet | ✅ | `lexical.py`, `SearchFilters`, `snippet.py` |
| 1 | Time-decay / QDF | ✅ | `ranking.py` (đã tinh chỉnh: query thường decay nhẹ, tin nóng recency áp đảo) |
| 1 | Đồng bộ gỡ bài (unpublish) | ✅ | `manager.remove_article` gỡ khỏi mọi chỉ mục |
| 1 | **Log click / dwell time** | ❌ | **Chưa có** — kế hoạch nhấn mạnh "từ ngày đầu"; cần ưu tiên |
| 2 | Embedding (hash/BGE-m3/OpenAI) | ✅ | `embeddings.py` — thêm `BGEEmbedder` (BGE-m3 local) |
| 2 | Semantic + RRF hybrid | ✅ | `vector.py`, `fusion.py`, `pipeline.py` |
| 2 | Milvus + **HNSW** | 🟡 | `milvus_vector.py` (HNSW, import-guard) — **cần Milvus server để chạy, chưa test offline** |
| 2 | Backfill / reindex job | 🟡 | `bulk_index` batch ✅; **chưa có job backfill riêng + blue-green reindex** |
| 3 | Rerank tinh (BGE-reranker) | ✅ | `rerank.py` (import-guard) |
| 3 | **Query rewrite / sửa lỗi / mở rộng** | ❌ | **Chưa có** — mới có nhận diện fresh-intent, chưa spell-correct/rewrite/expansion |
| 3 | Bộ đánh giá nDCG/MRR + latency | 🟡 | `benchmark.py` (Recall@10/MRR/nDCG@10 + p50/p95, tách lexical/dense/hybrid) ✅; **chưa có bộ nhãn lớn + LLM-as-judge** |
| 3 | Regression set ranking | ❌ | Test chức năng đóng vai regression; **chưa có regression chất lượng xếp hạng** |
| 4 | Dedup / gom cụm (MinHash-LSH) | ✅ | `dedup.py` |
| 4 | Đa dạng hóa (MMR) | ✅ | `diversify.py` |
| 4 | NER trích thực thể | ✅ | `entities.py` (heuristic + underthesea guard) |
| 4 | Knowledge graph Neo4j | 🟡 | KG **in-memory** ✅; **chưa nối Neo4j** |
| 4 | Phân loại intent (nav/info/fresh) | 🟡 | Mới có **fresh-intent nhị phân**; chưa phân loại đầy đủ |
| 5 | Dashboard (latency/CTR/index lag/zero-result) | ❌ | **Chưa có** |
| 5 | Cảnh báo (alert) | ❌ | **Chưa có** |
| 5 | **Cache truy vấn nóng** | ❌ | **Chưa có** (thiết kế có nêu — đòn bẩy chi phí quan trọng) |
| 5 | A/B / interleaving | ❌ | **Chưa có** |
| 5 | Reindex blue-green | ❌ | **Chưa có** |
| 5 | Review query 0 kết quả | 🟡 | Quan sát được (zero-result đo được); **chưa có công cụ/quy trình** |

### Nhóm thiếu quan trọng nhất (ưu tiên làm tiếp)

1. **Log click/feedback (GĐ1)** — tài sản không tạo lại được; nền tảng cho mọi đánh giá & học xếp hạng. Nên làm ngay.
2. **Query understanding thật (GĐ3)** — spell-correct + rewrite/expansion; hiện là điểm yếu rõ nhất cho truy vấn ngắn/mơ hồ.
3. **Backend production thật** — OpenSearch (chưa có), Milvus (có code, cần dựng server + test), Redis cache. Hiện toàn bộ **in-memory, không bền vững, không HA/snapshot**.
4. **Vận hành GĐ5** — monitoring/alert/cache/A-B/blue-green reindex: chưa khởi động.
5. **Phần từ BaoCao chưa nằm trong F-01..F-15**: autocomplete/gợi ý, phân trang (pagination) + facet counts, auth/rate-limit/observability.

### Đã bổ sung trong đợt này
- `BGEEmbedder` (BGE-m3 local, đa ngữ) — `EMBEDDER=bge`.
- `MilvusVectorIndex` với chỉ mục **HNSW** (`M`/`efConstruction`/`ef` cấu hình) — `VECTOR_BACKEND=milvus`.
- `benchmark.py` — so sánh hiệu quả (Recall@10/MRR/nDCG@10) & tốc độ (p50/p95) của lexical vs dense vs hybrid.

---

## 9. Cập nhật đợt 2 (2026-07-08) — bốn nhóm thiếu đã hiện thực

Tất cả là **tính năng phụ bật/tắt có trách nhiệm**: mặc định tắt/an toàn, degrade
khi thiếu phụ thuộc, không đổi hành vi lõi (146 test pass, 2 skipped).

| Trước | Hạng mục | Nay | Hiện thực |
|---|---|---|---|
| ❌ | **Log click/dwell (GĐ1)** | ✅ | `feedback/` — logger JSONL/Redis, `/search`→`X-Search-Id`, `POST /events/click`, `analytics.compute_metrics` (CTR/zero-result/position/dwell). Cờ `FEEDBACK_ENABLED` |
| ❌ | **Query understanding (GĐ3)** | ✅ | `understanding.py` — spell-correct Norvig trên vocab corpus + mở rộng đồng nghĩa + LLM rewrite (tùy chọn). Cờ `QU_SPELLCORRECT/QU_EXPANSION/QU_LLM_REWRITE` |
| 🟡 | **Backend production** | ✅/🟡 | OpenSearch BM25 (`opensearch_lexical.py` + factory `get_lexical_index`) ✅; Redis cho cache/feedback ✅; snapshot/restore JSONL (bền vững) ✅; Milvus vẫn cần server để test 🟡 |
| ❌ | **Cache truy vấn nóng (GĐ5)** | ✅ | `service/cache.py` — TTL+LRU memory / Redis, **vô hiệu theo generation**, bỏ qua tin nóng. Cờ `CACHE_ENABLED` |
| ❌ | **Dashboard + alert (GĐ5)** | ✅ | `service/metrics.py` — `/metrics` (Prometheus), `/dashboard` (HTML), cảnh báo ngưỡng p95/error/zero-result |
| ❌ | **Blue-green reindex (GĐ5)** | ✅ | `SearchService.reindex` (build mới → hoán đổi nguyên tử) + `POST /admin/reindex` (bảo vệ `ADMIN_TOKEN`) |
| ❌ | **A/B (GĐ5)** | 🟡 | Khung A/B: gán biến thể `X-Variant` + log để phân tích. Interleaving thực thụ: chưa |

**Còn lại (ngoài phạm vi đợt này):** interleaving online đầy đủ, LLM-as-judge cho
bộ đánh giá, học xếp hạng (LTR) từ log click, và kiểm thử Milvus/OpenSearch/Redis
trên server thật (hiện có code + import-guard, chưa có server trong môi trường dev).
