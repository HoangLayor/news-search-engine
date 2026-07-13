"""Vector backend production: Milvus với chỉ mục HNSW và Sparse (Hybrid Search).

Cùng interface với :class:`news_search.index.vector.VectorIndex` nhưng mở rộng để
chứa cả BM25 (sparse vector) và metadata (title, body, url, ...).
"""

from __future__ import annotations

import json
import logging
import hashlib
from collections import Counter
from datetime import datetime
import numpy as np

from news_search.config import Settings
from news_search.ingest.tokenizer import fold_tokens, tokenize
from news_search.models import Article

_log = logging.getLogger(__name__)


def _text_to_sparse(text: str) -> dict[int, float]:
    """Chuyển đổi văn bản thành sparse vector dựa trên tần suất từ (TF)."""
    tokens = fold_tokens(tokenize(text))
    tf = Counter(tokens)
    out = {}
    for token, count in tf.items():
        # Hash token thành số nguyên dương 32-bit (Milvus sparse vector keys)
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
        if h in out:
            out[h] += float(count)
        else:
            out[h] = float(count)
    return out


class MilvusVectorIndex:
    """Chỉ mục Hybrid trên Milvus (Dense HNSW + Sparse BM25 + Metadata)."""

    def __init__(self, dim: int, settings: Settings) -> None:
        if dim <= 0:
            raise ValueError("dim phải > 0")
        try:
            from pymilvus import DataType, MilvusClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("MilvusVectorIndex cần package 'pymilvus' (chưa cài)") from exc

        self.dim = dim
        self.collection = settings.milvus_collection
        self.metric = settings.milvus_metric.upper()
        self._ef_search = settings.hnsw_ef_search
        self.settings = settings

        try:
            token = settings.milvus_token or None
            self._client = MilvusClient(uri=settings.milvus_uri, token=token)
            self._ensure_collection(DataType, settings)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Không kết nối/khởi tạo được Milvus: {exc}") from exc

    # ------------------------------------------------------------------ setup
    def _existing_dim(self) -> int | None:
        try:
            desc = self._client.describe_collection(self.collection)
            for f in desc.get("fields", []):
                if f.get("name") in ("dense_vector", "vector"):
                    dim = (f.get("params") or {}).get("dim")
                    if dim is not None:
                        return int(dim)
        except Exception:
            pass
        return None

    def _create_collection(self, DataType, settings: Settings) -> None:
        client = self._client
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("article_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("metadata", DataType.JSON)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type=self.metric,
            params={"M": settings.hnsw_m, "efConstruction": settings.hnsw_ef_construction},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_collection(self, DataType, settings: Settings) -> None:
        client = self._client
        if client.has_collection(self.collection):
            desc = client.describe_collection(self.collection)
            fields = {f.get("name") for f in desc.get("fields", [])}
            has_new_fields = {"dense_vector", "sparse_vector", "metadata"}.issubset(fields)
            
            existing = self._existing_dim()
            if not has_new_fields or (existing is not None and existing != self.dim):
                if settings.milvus_recreate_on_dim_mismatch or not has_new_fields:
                    _log.warning(f"  [CẢNH BÁO] Collection {self.collection} dùng schema cũ hoặc sai dim. Đang recreate...")
                    client.drop_collection(self.collection)
                    self._create_collection(DataType, settings)
                else:
                    raise RuntimeError(
                        f"Collection '{self.collection}' đang có dim={existing} hoặc sai schema, nhưng "
                        f"embedder hiện tại yêu cầu dim={self.dim}.\n"
                        "-> HÃY BẬT cờ MILVUS_RECREATE_ON_DIM_MISMATCH=true để cho phép "
                        "xóa collection cũ và tạo lại."
                    )
        else:
            self._create_collection(DataType, settings)
        client.load_collection(self.collection)

    def _normalized(self, vector: np.ndarray) -> list[float]:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.tolist()
        
    def _article_to_metadata(self, article: Article) -> dict:
        import dataclasses
        meta = dataclasses.asdict(article)
        # Truncate body if it exceeds reasonable JSON size, normally unnecessary but safe
        if len(meta.get("body", "")) > 60000:
            meta["body"] = meta["body"][:60000]
        # published_at must be string in JSON
        if isinstance(meta.get("published_at"), datetime):
            meta["published_at"] = meta["published_at"].isoformat()
        return meta
        
    def _metadata_to_article(self, meta: dict) -> Article:
        if isinstance(meta.get("published_at"), str):
            try:
                meta["published_at"] = datetime.fromisoformat(meta["published_at"])
            except ValueError:
                meta["published_at"] = datetime.now() # fallback
        return Article(**meta)

    # -------------------------------------------------------------------- API Store Mới
    def get_article(self, article_id: str) -> Article | None:
        """Lấy bài viết gốc từ JSON metadata."""
        rows = self._client.get(
            self.collection, ids=[article_id], output_fields=["metadata"]
        )
        if not rows:
            return None
        return self._metadata_to_article(rows[0]["metadata"])

    def mget_articles(self, article_ids: list[str]) -> dict[str, Article]:
        if not article_ids:
            return {}
        rows = self._client.get(
            self.collection, ids=article_ids, output_fields=["metadata"]
        )
        out = {}
        for r in rows:
            aid = r.get("id") or r.get("article_id")
            if not aid and r.get("metadata"):
                aid = r["metadata"].get("article_id")
            if aid and r.get("metadata"):
                out[aid] = self._metadata_to_article(r["metadata"])
        return out

    # -------------------------------------------------------------------- API
    def add(self, article_id: str, vector: np.ndarray) -> None:
        raise NotImplementedError("Sử dụng add_hybrid thay thế.")

    def add_hybrid(self, article: Article, vector: np.ndarray) -> None:
        """Upsert bài viết với cả dense và sparse vector."""
        # Trọng số tiêu đề được nhân lên 3 lần
        text_for_sparse = " ".join([article.title] * 3) + " " + article.body
        sparse_vec = _text_to_sparse(text_for_sparse)
        if not sparse_vec: # sparse vector cannot be empty
            sparse_vec = {0: 1.0}
            
        data = [{
            "article_id": article.article_id,
            "dense_vector": self._normalized(vector),
            "sparse_vector": sparse_vec,
            "metadata": self._article_to_metadata(article)
        }]
        self._client.upsert(self.collection, data=data)

    def add_batch_hybrid(self, articles: list[Article], vectors: list[np.ndarray]) -> None:
        if not articles:
            return
        data = []
        for art, vec in zip(articles, vectors):
            text_for_sparse = " ".join([art.title] * 3) + " " + art.body
            sparse_vec = _text_to_sparse(text_for_sparse)
            if not sparse_vec: sparse_vec = {0: 1.0}
            data.append({
                "article_id": art.article_id,
                "dense_vector": self._normalized(vec),
                "sparse_vector": sparse_vec,
                "metadata": self._article_to_metadata(art)
            })
        self._client.upsert(self.collection, data=data)

    def remove(self, article_id: str) -> None:
        self._client.delete(self.collection, ids=[article_id])

    def get(self, article_id: str) -> np.ndarray | None:
        rows = self._client.get(self.collection, ids=[article_id], output_fields=["dense_vector"])
        if not rows:
            return None
        return np.asarray(rows[0]["dense_vector"], dtype=np.float32)

    def _build_filter(self, allowed_ids: set[str] | None) -> str:
        if allowed_ids is not None:
            if not allowed_ids:
                return "article_id == 'NONE'"  # impossible
            ids_literal = ", ".join(f'"{i}"' for i in allowed_ids)
            return f"article_id in [{ids_literal}]"
        return ""

    def search_semantic(self, vector: np.ndarray, top_k: int = 10, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if top_k <= 0: return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(q)) == 0.0: return []
        
        res = self._client.search(
            self.collection,
            data=[self._normalized(q)],
            anns_field="dense_vector",
            limit=top_k,
            filter=self._build_filter(allowed_ids),
            search_params={"metric_type": self.metric, "params": {"ef": self._ef_search}},
            output_fields=["article_id"]
        )
        hits = res[0] if res else []
        return [(h.get("id") or h.get("entity", {}).get("article_id"), float(h["distance"])) for h in hits]

    def search_lexical(self, query_text: str, top_k: int = 10, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if top_k <= 0 or not query_text.strip(): return []
        q_sparse = _text_to_sparse(query_text)
        if not q_sparse: return []

        res = self._client.search(
            self.collection,
            data=[q_sparse],
            anns_field="sparse_vector",
            limit=top_k,
            filter=self._build_filter(allowed_ids),
            search_params={"metric_type": "IP"},
            output_fields=["article_id"]
        )
        hits = res[0] if res else []
        return [(h.get("id") or h.get("entity", {}).get("article_id"), float(h["distance"])) for h in hits]

    def search_hybrid(self, query_text: str, vector: np.ndarray, top_k: int = 10, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if top_k <= 0: return []
        
        try:
            from pymilvus import AnnSearchRequest, RRFRanker
        except ImportError:
            # Fallback to pure semantic if Milvus version < 2.4
            return self.search_semantic(vector, top_k, allowed_ids)
            
        q_dense = self._normalized(vector)
        q_sparse = _text_to_sparse(query_text)
        
        req_dense = AnnSearchRequest(
            data=[q_dense],
            anns_field="dense_vector",
            param={"metric_type": self.metric, "params": {"ef": self._ef_search}},
            limit=top_k * 2
        )
        req_sparse = AnnSearchRequest(
            data=[q_sparse] if q_sparse else [{0: 1.0}], # dummy if empty
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k * 2
        )
        
        res = self._client.hybrid_search(
            self.collection,
            reqs=[req_sparse, req_dense],
            ranker=RRFRanker(k=60), # k=60 as standard RRF
            limit=top_k,
            filter=self._build_filter(allowed_ids),
            output_fields=["article_id"]
        )
        hits = res[0] if res else []
        return [(h.get("id") or h.get("entity", {}).get("article_id"), float(h["distance"])) for h in hits]

    def __len__(self) -> int:
        try:
            rows = self._client.query(self.collection, filter="", output_fields=["count(*)"])
            return int(rows[0]["count(*)"]) if rows else 0
        except Exception:
            return 0
