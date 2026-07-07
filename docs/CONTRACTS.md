# CONTRACTS — Đặc tả interface các module (BẮT BUỘC tuân thủ)

> Tài liệu này là hợp đồng giữa các module. Mọi implementer PHẢI giữ đúng tên class,
> tên hàm, chữ ký (signature) và hành vi mô tả ở đây. Model dữ liệu chung ở
> `news_search/models.py`, cấu hình ở `news_search/config.py` — ĐÃ VIẾT SẴN, không sửa.
> Ánh xạ chức năng F-01..F-15: xem `BaoCao-ThietKe.md` §2.

Quy tắc chung:
- Python 3.11, type hints đầy đủ, docstring tiếng Việt ngắn gọn.
- Core modules chỉ dùng **stdlib + numpy**. Thư viện nặng (underthesea, openai,
  FlagEmbedding, opensearch-py, pymilvus, neo4j) phải nằm sau **import guard**
  (try/except ImportError) và có fallback thuần Python.
- **Determinism**: mọi hashing phải deterministic giữa các process — dùng
  `hashlib`/`zlib.crc32`, KHÔNG dùng `hash()` built-in (bị randomize).
- Mọi `datetime` là timezone-aware. So sánh naive/aware sẽ raise — cứ để raise.
- Text tiếng Việt: chuẩn hóa NFC; matching không phân biệt dấu dùng bản "folded".

---

## 1. `news_search/ingest/tokenizer.py` (F-03)

```python
def normalize_text(text: str) -> str
    # NFC, lowercase, gọn khoảng trắng (mọi whitespace -> 1 space, strip)

def fold_diacritics(text: str) -> str
    # Bỏ dấu tiếng Việt: NFD + bỏ combining marks; "đ"->"d", "Đ"->"d".
    # KHÔNG lowercase (hàm này chỉ bỏ dấu; caller tự normalize trước nếu cần).

def tokenize(text: str) -> list[str]
    # normalize_text -> tách token: giữ chữ + số, bỏ dấu câu.
    # Nếu underthesea import được: dùng word_tokenize (ghép từ ghép bằng "_")
    # sau khi normalize; nếu không: split theo regex \w+ (unicode).
    # Trả về list token đã lowercase, KHÔNG bỏ dấu.

def fold_tokens(tokens: list[str]) -> list[str]
    # fold_diacritics từng token (dùng cho matching không dấu).
```

## 2. `news_search/ingest/cleaner.py` (F-02)

```python
def clean_html(html: str) -> str
    # Bỏ <script>/<style> cùng nội dung, bỏ mọi tag, unescape HTML entities,
    # gọn khoảng trắng nhưng GIỮ ngắt đoạn (đoạn <p>, <br>, <div> -> "\n").
    # Thuần regex + html.unescape (stdlib) — KHÔNG cần bs4.

def normalize_article(raw: dict) -> Article
    # Nhận dict thô từ CMS/API -> Article đã chuẩn hóa:
    # - Thiếu field bắt buộc (models.REQUIRED_FIELDS) -> raise ValueError, message nêu rõ field.
    # - body: nếu chứa tag HTML thì clean_html.
    # - published_at: chấp nhận datetime hoặc chuỗi ISO-8601;
    #   chuỗi không có timezone -> gán UTC+7 (Asia/Ho_Chi_Minh, dùng
    #   datetime.timezone(timedelta(hours=7))). datetime naive -> gán UTC+7.
    # - title/author/category/source: strip khoảng trắng thừa.
    # - status mặc định "published"; tags mặc định [].
```

## 3. `news_search/index/lexical.py` (F-04, F-09) — BM25 local

```python
class LexicalIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75, title_weight: int = 3): ...
    def add(self, article: Article) -> None      # re-add cùng id = replace, không trùng lặp
    def update(self, article: Article) -> None   # = remove + add
    def remove(self, article_id: str) -> None    # id không tồn tại -> no-op
    def search(self, query_text: str, top_k: int = 10,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]
        # BM25 trên token FOLDED (không dấu, lowercase) để match không phân biệt dấu.
        # Doc tokens = fold_tokens(tokenize(title)) * title_weight + fold_tokens(tokenize(body)).
        # Query tokens = fold_tokens(tokenize(query_text)).
        # allowed_ids != None -> chỉ chấm điểm doc trong tập đó.
        # Trả list (article_id, score>0) sắp giảm dần theo score. Query rỗng -> [].
    def __contains__(self, article_id: str) -> bool
    def __len__(self) -> int
```
- Inverted index: token -> {doc_id: term_freq}. IDF theo công thức BM25 chuẩn
  (Robertson): `ln((N - df + 0.5)/(df + 0.5) + 1)`.
- Import tokenizer từ `news_search.ingest.tokenizer`.

## 4. `news_search/search/snippet.py` (F-15)

```python
def make_snippet(body: str, query_tokens: list[str], max_len: int = 200) -> str
    # Chọn cửa sổ ~max_len ký tự chứa NHIỀU token khớp nhất (match không dấu,
    # không phân biệt hoa thường, so trên từ nguyên vẹn).
    # Bôi đậm từ khớp bằng <b>...</b> (giữ nguyên chữ gốc CÓ DẤU của body).
    # Thêm "..." đầu/cuối nếu cắt giữa văn bản. Không match nào -> đầu body cắt max_len.
    # body rỗng -> "".
```

## 5. `news_search/index/embeddings.py` (F-05)

```python
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray  # shape (n, dim), L2-normalized, float32

class HashingEmbedder:  # fallback local, deterministic, offline
    def __init__(self, dim: int = 256): ...
    # Char n-gram (n=2..4) trên text đã normalize+fold; feature hashing bằng
    # zlib.crc32 -> bucket = crc % dim, sign = bit khác của crc; tf-weight rồi L2 normalize.
    # Vector zero (text rỗng) -> trả zero vector, KHÔNG chia 0.

class OpenAIEmbedder:  # import guard; thiếu package/key -> raise RuntimeError khi khởi tạo
    def __init__(self, model: str, api_key: str, dim: int = 3072): ...

def get_embedder(settings: Settings) -> Embedder
    # settings.embedder == "hash" -> HashingEmbedder(settings.embedding_dim)
    # == "openai" -> OpenAIEmbedder(...)  ; giá trị khác -> ValueError
```

## 6. `news_search/index/vector.py` (F-05, F-10) — ANN local (brute-force cosine)

```python
class VectorIndex:
    def __init__(self, dim: int): ...
    def add(self, article_id: str, vector: np.ndarray) -> None  # replace nếu trùng id; lưu bản L2-normalized
    def remove(self, article_id: str) -> None                   # no-op nếu không có
    def get(self, article_id: str) -> np.ndarray | None         # vector đã normalize
    def search(self, vector: np.ndarray, top_k: int = 10,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]
        # Cosine similarity, sắp giảm dần. Query vector zero -> [].
    def __len__(self) -> int
```
- Brute-force bằng numpy matrix (ma trận (n, dim) + mapping id/row; xoá = đánh dấu
  hoặc rebuild — miễn đúng hành vi). Ghi chú docstring: production thay bằng Milvus/HNSW.

## 7. `news_search/search/query.py` (Query Understanding)

```python
FRESH_HINT_TOKENS: frozenset[str]
    # dạng FOLDED, tối thiểu: hom nay, moi nhat, moi, vua, vua xay ra, dang,
    # truc tiep, nong, breaking, sang nay, chieu nay, toi nay, tuan nay, live
    # (lưu từng token đơn; cụm nhiều từ kiểm tra bằng substring trên chuỗi folded)

def parse_query(text: str) -> ParsedQuery
    # text rỗng/toàn whitespace -> raise ValueError("Truy vấn rỗng")
    # normalized = tokenizer.normalize_text(text); folded = fold_diacritics(normalized)
    # tokens = tokenize(text); fresh_intent = có hint token/cụm trong folded
```

## 8. `news_search/search/fusion.py` (F-11 — RRF)

```python
def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]
    # score(id) = sum over rankings chứa id của 1/(k + rank), rank bắt đầu = 1.
    # Ranking rỗng bỏ qua. Trả dict {id: score}.
```

## 9. `news_search/search/ranking.py` (F-13 — time-decay/QDF)

```python
def time_decay_multiplier(published_at: datetime, now: datetime,
                          half_life_days: float, floor: float = 0.3) -> float
    # age_days = max(0, (now - published_at).total_seconds()/86400)
    # return floor + (1 - floor) * 0.5 ** (age_days / half_life_days)
    # half_life_days <= 0 -> trả 1.0 (tắt decay).

def apply_time_decay(scores: dict[str, float],
                     published_lookup: Callable[[str], datetime],
                     now: datetime, half_life_days: float,
                     floor: float = 0.3) -> dict[str, float]
    # Nhân từng score với multiplier tương ứng, trả dict mới.
```

## 10. `news_search/search/rerank.py` (rerank tinh — GĐ 3)

```python
class Reranker(Protocol):
    def rerank(self, query: str, docs: list[tuple[str, str]],
               top_k: int | None = None) -> list[tuple[str, float]]
        # docs = [(article_id, doc_text)] theo thứ tự hiện tại.
        # Trả [(article_id, score)] sắp giảm dần theo score mới.

class NoopReranker:  # giữ nguyên thứ tự vào, score = 1/(rank)
class BGEReranker:   # import guard FlagEmbedding; thiếu -> raise RuntimeError khi __init__

def get_reranker(settings: Settings) -> Reranker
    # "none" -> NoopReranker(); "bge" -> BGEReranker(); khác -> ValueError
```

## 11. `news_search/index/dedup.py` (F-07 — MinHash-LSH, thuần Python)

```python
class MinHashDeduper:
    def __init__(self, num_perm: int = 128, threshold: float = 0.7,
                 shingle_size: int = 3): ...
    def add(self, article_id: str, text: str) -> str
        # Shingle = word n-gram (n=shingle_size) trên fold_tokens(tokenize(text)).
        # MinHash num_perm hàm (xáo trộn bằng (a*h + b) % prime, seed cố định).
        # LSH banding (bands*rows == num_perm, chọn bands theo threshold ~ (1/b)^(1/r)).
        # Tìm ứng viên qua LSH, xác nhận bằng estimated jaccard >= threshold
        # -> gán vào cluster của ứng viên giống nhất; không có -> cluster mới
        # (cluster_id = article_id đầu tiên của cụm). Trả cluster_id.
        # add lại cùng article_id -> remove trước rồi add (không trùng lặp).
    def remove(self, article_id: str) -> None
        # Gỡ khỏi mọi cấu trúc; cluster_id của các thành viên còn lại GIỮ NGUYÊN.
    def cluster_of(self, article_id: str) -> str | None
    def similarity(self, id_a: str, id_b: str) -> float  # estimated jaccard; thiếu id -> 0.0
    def __len__(self) -> int
```

## 12. `news_search/search/diversify.py` (F-14 — collapse cụm + MMR)

```python
def collapse_clusters(ranked_ids: list[str],
                      cluster_of: Callable[[str], str | None],
                      max_per_cluster: int = 1) -> list[str]
    # Duyệt theo thứ tự, mỗi cluster giữ tối đa max_per_cluster id.
    # cluster_of trả None -> coi như cụm riêng (không gộp).

def mmr(ranked: list[tuple[str, float]],
        vector_lookup: Callable[[str], np.ndarray | None],
        lambda_: float = 0.7, top_k: int = 10) -> list[str]
    # Maximal Marginal Relevance: chọn lặp id có
    #   lambda_*rel_norm - (1-lambda_)*max_sim_với_đã_chọn  lớn nhất.
    # rel_norm = score chuẩn hóa min-max về [0,1] (mọi score bằng nhau -> 1.0).
    # vector_lookup trả None -> coi sim = 0 với mọi doc (vẫn chọn được theo rel).
    # Trả list id theo thứ tự chọn, tối đa top_k.
```

## 13. `news_search/index/entities.py` (F-06 — NER + knowledge graph local)

```python
def extract_entities(text: str) -> list[Entity]
    # underthesea.ner import được -> dùng, map nhãn (PER/ORG/LOC; khác -> MISC).
    # Fallback heuristic: cụm >=1 từ viết hoa liên tiếp (tôn trọng chữ có dấu),
    # bỏ từ đầu câu đơn lẻ nếu là stopword thường gặp (Ông, Bà, Theo, Tại, Trong...),
    # cụm chứa các từ như (Công ty, Tập đoàn, Bộ, Sở, UBND, Trường, Ngân hàng -> ORG;
    # tỉnh/thành/quận/huyện/phường/xã hoặc tên sau "tại", "ở" -> LOC; còn lại PER nếu
    # 2-4 từ viết hoa, không thì MISC). Không cần hoàn hảo — cần deterministic + có test.
    # Dedup theo (name, type), giữ thứ tự xuất hiện.

class KnowledgeGraph:  # in-memory; production: Neo4j
    def add_article(self, article_id: str, entities: list[Entity]) -> None  # re-add = replace
    def remove_article(self, article_id: str) -> None
    def entities_for_article(self, article_id: str) -> list[Entity]
    def articles_for_entity(self, name: str) -> set[str]   # match không dấu, không hoa/thường
    def top_entities(self, n: int = 10) -> list[tuple[Entity, int]]  # theo số bài
```

## 14. `news_search/index/manager.py` (F-01, F-08 — integrator viết)

```python
class ArticleStore:
    def put(self, article: Article) -> None
    def get(self, article_id: str) -> Article | None
    def delete(self, article_id: str) -> None
    def filter_ids(self, filters: SearchFilters) -> set[str] | None  # None nếu filters rỗng
    def __len__(self) -> int

class IndexManager:
    def __init__(self, settings: Settings | None = None): ...
    # thuộc tính: settings, store, lexical, vector, embedder, deduper, kg
    def index_article(self, raw: dict | Article) -> Article
        # dict -> normalize_article. status != "published" -> remove_article + return.
        # Ghi vào: store, lexical, vector (embed article.text), deduper, kg. Re-index an toàn.
    def bulk_index(self, raws: list[dict | Article]) -> int
    def remove_article(self, article_id: str) -> None  # gỡ khỏi MỌI chỉ mục (F-08)
```

## 15. `news_search/search/pipeline.py` (integrator viết)

```python
class SearchPipeline:
    def __init__(self, manager: IndexManager, settings: Settings | None = None): ...
    def search(self, query: SearchQuery) -> list[SearchResultItem]
        # 1. parse_query (ValueError nếu rỗng — cứ để propagate)
        # 2. allowed_ids = store.filter_ids(query.filters)
        # 3. mode lexical/semantic/hybrid -> lấy candidates_k từ mỗi nhánh
        # 4. hybrid: rrf 2 ranking; đơn nhánh: dùng score nhánh đó
        # 5. now = query.now or datetime.now(timezone.utc);
        #    half_life = fresh_half_life_days nếu fresh_intent else half_life_days
        #    apply_time_decay
        # 6. reranker (NoopReranker mặc định) trên top rerank_top_n
        # 7. collapse_clusters (deduper.cluster_of, max_per_cluster)
        # 8. mmr trên top mmr_pool -> lấy top_k
        # 9. make_snippet(article.body, parsed.tokens) -> SearchResultItem
```

## 16. `news_search/api/app.py` (integrator viết — FastAPI)

- `create_app(settings=None) -> FastAPI`; module-level `app = create_app()`.
- `POST /articles` (1 bài, JSON) → 201 `{"indexed": true, "article_id": ...}`;
  thiếu field bắt buộc → 422 với message rõ.
- `POST /articles/bulk` (list) → `{"indexed": n}`.
- `DELETE /articles/{id}` → 200 `{"removed": true}` (F-08).
- `GET /search?q=...&mode=&top_k=&author=&category=&source=&date_from=&date_to=`
  → JSON **array** các object ĐÚNG 5 trường. `q` rỗng/thiếu → 400 message rõ ràng.
- `GET /articles/{id}/entities` → list entity. `GET /healthz` → counts.

---

## Môi trường & quy trình cho MỌI agent

- CWD: `D:\MEDIASOFT\news-search-engine`. Code UTF-8.
- Python venv: `D:\MEDIASOFT\news-search-engine\.venv\Scripts\python.exe`
  (đã cài fastapi, uvicorn, pydantic v2, numpy, pytest, httpx — KHÔNG pip install thêm).
- Chạy test:
  `& "D:\MEDIASOFT\news-search-engine\.venv\Scripts\python.exe" -m pytest tests/<file> -q`
- CHỈ tạo/sửa đúng các file được giao. KHÔNG sửa models.py, config.py, CONTRACTS.md,
  file của agent khác.
- Test phải deterministic (không phụ thuộc thời gian thực — truyền `now` cố định).
