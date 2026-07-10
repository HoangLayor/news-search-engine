"""Vector backend production: Milvus với chỉ mục HNSW (F-05, F-10).

Cùng interface với :class:`news_search.index.vector.VectorIndex` để thay thế
trong suốt qua ``VECTOR_BACKEND=milvus``. Dùng ``MilvusClient`` (pymilvus >= 2.4).

Chỉ mục **HNSW** với tham số từ Settings:
    index_type = "HNSW"
    metric_type = MILVUS_METRIC (mặc định COSINE)
    params = { "M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION }
truy vấn dùng { "ef": HNSW_EF_SEARCH }.

Import guard: thiếu ``pymilvus`` HOẶC không kết nối được -> ``RuntimeError`` ngay
khi ``__init__`` (fail-fast, không im lặng rơi về local).

LƯU Ý: file này cần một Milvus server đang chạy để hoạt động; không có test
offline. Chạy Milvus nhanh:  ``docker run -p 19530:19530 milvusdb/milvus:latest``
hoặc dùng Milvus Lite (``MILVUS_URI=./milvus.db``).
"""

from __future__ import annotations

import numpy as np

from news_search.config import Settings


class MilvusVectorIndex:
    """Chỉ mục vector trên Milvus + HNSW, khớp interface của VectorIndex."""

    def __init__(self, dim: int, settings: Settings) -> None:
        if dim <= 0:
            raise ValueError("dim phải > 0")
        try:
            from pymilvus import DataType, MilvusClient  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "MilvusVectorIndex cần package 'pymilvus' (chưa cài) — "
                "cài pymilvus hoặc dùng VECTOR_BACKEND=local"
            ) from exc

        self.dim = dim
        self.collection = settings.milvus_collection
        self.metric = settings.milvus_metric.upper()
        self._ef_search = settings.hnsw_ef_search

        try:
            token = settings.milvus_token or None
            self._client = MilvusClient(uri=settings.milvus_uri, token=token)
            self._ensure_collection(DataType, settings)
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - lỗi kết nối/thao tác server
            raise RuntimeError(f"Không kết nối/khởi tạo được Milvus: {exc}") from exc

    # ------------------------------------------------------------------ setup
    def _existing_dim(self) -> int | None:
        """Số chiều của trường vector trong collection đã tồn tại (None nếu không rõ)."""
        desc = self._client.describe_collection(self.collection)
        for f in desc.get("fields", []):
            dim = (f.get("params") or {}).get("dim")
            if dim is not None:
                return int(dim)
        return None

    def _create_collection(self, DataType, settings: Settings) -> None:
        """Tạo collection mới với chỉ mục HNSW theo cấu hình."""
        client = self._client
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("article_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",  # <-- HNSW
            metric_type=self.metric,
            params={
                "M": settings.hnsw_m,
                "efConstruction": settings.hnsw_ef_construction,
            },
        )
        client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_collection(self, DataType, settings: Settings) -> None:
        """Tạo collection + chỉ mục HNSW nếu chưa có; nạp vào bộ nhớ để search.

        Nếu collection ĐÃ TỒN TẠI với dim KHÁC embedder hiện tại: mặc định raise
        lỗi rõ ràng (nếu không, Milvus sẽ từ chối MỌI insert và — do reindex nuốt
        exception — bạn chỉ thấy "0 bài" mà không biết vì sao). Đặt
        ``MILVUS_RECREATE_ON_DIM_MISMATCH=true`` để tự động drop + tạo lại.
        """
        client = self._client
        if client.has_collection(self.collection):
            existing = self._existing_dim()
            if existing is not None and existing != self.dim:
                if not settings.milvus_recreate_on_dim_mismatch:
                    raise RuntimeError(
                        f"Collection '{self.collection}' có dim={existing} nhưng embedder "
                        f"({settings.embedder}) sinh vector dim={self.dim}. Milvus sẽ từ chối "
                        f"mọi insert. Cách xử lý: (a) đổi EMBEDDER cho khớp dim={existing}, "
                        f"(b) đổi MILVUS_COLLECTION sang tên mới, hoặc "
                        f"(c) đặt MILVUS_RECREATE_ON_DIM_MISMATCH=true để DROP + tạo lại "
                        f"(XÓA toàn bộ vector cũ)."
                    )
                client.drop_collection(self.collection)
                self._create_collection(DataType, settings)
        else:
            self._create_collection(DataType, settings)
        client.load_collection(self.collection)

    # ------------------------------------------------------------------ utils
    def _normalized(self, vector: np.ndarray) -> list[float]:
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Vector dim {vec.shape[0]} không khớp {self.dim}")
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.tolist()

    # -------------------------------------------------------------------- API
    def add(self, article_id: str, vector: np.ndarray) -> None:
        """Upsert (replace nếu trùng id). Lưu bản đã L2-normalize."""
        self._client.upsert(
            self.collection,
            data=[{"article_id": article_id, "vector": self._normalized(vector)}],
        )

    def remove(self, article_id: str) -> None:
        """Xóa theo id; không tồn tại -> no-op."""
        self._client.delete(self.collection, ids=[article_id])

    def get(self, article_id: str) -> np.ndarray | None:
        """Lấy vector đã lưu (phục vụ MMR); không có -> None."""
        rows = self._client.get(
            self.collection, ids=[article_id], output_fields=["vector"]
        )
        if not rows:
            return None
        return np.asarray(rows[0]["vector"], dtype=np.float32)

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Tìm top_k láng giềng HNSW theo metric cấu hình; sắp giảm dần.

        - ``allowed_ids`` != None -> lọc bằng biểu thức ``article_id in [...]``.
          (tập rỗng -> trả []). Query vector zero -> [].
        """
        if top_k <= 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(q)) == 0.0:
            return []
        filter_expr = ""
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            ids_literal = ", ".join(f'"{i}"' for i in allowed_ids)
            filter_expr = f"article_id in [{ids_literal}]"

        res = self._client.search(
            self.collection,
            data=[self._normalized(q)],
            limit=top_k,
            filter=filter_expr,
            search_params={"metric_type": self.metric, "params": {"ef": self._ef_search}},
            output_fields=["article_id"],
        )
        hits = res[0] if res else []
        out: list[tuple[str, float]] = []
        for h in hits:
            aid = h.get("id") or h.get("entity", {}).get("article_id")
            score = float(h["distance"])  # COSINE/IP: lớn hơn = gần hơn
            out.append((aid, score))
        return out

    def __len__(self) -> int:
        stats = self._client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))
