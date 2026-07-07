"""Sinh embedding ngữ nghĩa (F-05) — CONTRACTS.md §5.

Cung cấp:
- ``Embedder`` (Protocol): giao diện chung — ``embed(texts) -> (n, dim) float32``,
  từng hàng đã L2-normalize.
- ``HashingEmbedder``: fallback local, deterministic, chạy offline — char n-gram
  (n=2..4) trên text đã normalize+fold, feature hashing bằng ``zlib.crc32``.
- ``OpenAIEmbedder``: gọi API OpenAI, nằm sau import guard — thiếu package hoặc
  thiếu api_key -> ``RuntimeError`` ngay khi ``__init__``.
- ``get_embedder(settings)``: factory chọn embedder theo cấu hình.

LƯU Ý phụ thuộc: normalize + fold được viết NỘI BỘ tối giản bằng ``unicodedata``
(hàm ``_normalize_fold``), KHÔNG import ``news_search.ingest.tokenizer`` để
tránh phụ thuộc chéo giữa các module.
"""

from __future__ import annotations

import unicodedata
import zlib
from typing import Protocol

import numpy as np

from news_search.config import Settings

# Kích thước char n-gram dùng cho feature hashing
_NGRAM_SIZES = (2, 3, 4)


def _normalize_fold(text: str) -> str:
    """Chuẩn hóa + bỏ dấu tối giản (nội bộ, không dùng ingest.tokenizer).

    Bước: NFC -> lowercase -> gọn khoảng trắng (mọi whitespace -> 1 space,
    strip) -> "đ"/"Đ" -> "d" -> NFD + bỏ combining marks (bỏ dấu tiếng Việt).
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = " ".join(text.split())
    # "đ" không decompose được qua NFD nên thay thủ công ("Đ" đã lowercase ở trên)
    text = text.replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


class Embedder(Protocol):
    """Giao diện chung cho mọi embedder (structural typing)."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả ma trận shape (n, dim), dtype float32, từng hàng L2-normalized."""
        ...


class HashingEmbedder:
    """Embedder local deterministic, offline (fallback khi không có API).

    Cách hoạt động:
    - Text được normalize + fold (``_normalize_fold``).
    - Sinh char n-gram với n = 2..4, đếm term frequency (tf).
    - Feature hashing: ``crc = zlib.crc32(ngram)``; bucket = ``crc % dim``;
      sign lấy từ BIT KHÁC của crc (bit 31 — độc lập với các bit thấp dùng
      cho bucket) -> cộng ``sign * tf`` vào bucket.
    - L2 normalize kết quả; text rỗng/quá ngắn -> zero vector (KHÔNG chia 0).

    Deterministic giữa các process vì chỉ dùng ``zlib.crc32`` (cấm ``hash()``).
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim phải > 0")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Vector hóa danh sách văn bản -> (n, dim) float32, L2-normalized."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            folded = _normalize_fold(text)
            # Đếm tf của từng char n-gram (n = 2..4)
            counts: dict[str, int] = {}
            for n in _NGRAM_SIZES:
                for i in range(len(folded) - n + 1):
                    gram = folded[i : i + n]
                    counts[gram] = counts.get(gram, 0) + 1
            vec = out[row]
            for gram, tf in counts.items():
                crc = zlib.crc32(gram.encode("utf-8"))
                bucket = crc % self.dim
                sign = 1.0 if (crc >> 31) & 1 == 0 else -1.0
                vec[bucket] += sign * tf
            # L2 normalize, zero-safe: vector zero giữ nguyên, không chia 0
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec /= norm
        return out


class OpenAIEmbedder:
    """Embedder production gọi OpenAI Embeddings API.

    Import guard: thiếu package ``openai`` HOẶC thiếu ``api_key`` ->
    ``RuntimeError`` ngay khi ``__init__`` (fail-fast, không đợi tới lúc embed).
    """

    def __init__(self, model: str, api_key: str, dim: int = 3072) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "Thiếu package 'openai' — cài package hoặc dùng EMBEDDER=hash"
            ) from exc
        if not api_key:
            raise RuntimeError(
                "Thiếu OPENAI_API_KEY — cấu hình key hoặc dùng EMBEDDER=hash"
            )
        self.dim = dim
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - cần API
        """Gọi API rồi L2-normalize kết quả về (n, dim) float32."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dim
        )
        arr = np.asarray([item.embedding for item in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        np.divide(arr, norms, out=arr, where=norms > 0)  # zero-safe
        return arr


class BGEEmbedder:
    """BGE-m3 local (BAAI/bge-m3) — dense đa ngữ 1024 chiều, hỗ trợ tốt tiếng Việt.

    Import guard hai tầng: ưu tiên ``FlagEmbedding.BGEM3FlagModel`` (native, còn
    có sparse/ColBERT nếu cần mở rộng), fallback ``sentence-transformers``. Thiếu
    cả hai -> ``RuntimeError`` ngay khi ``__init__`` (fail-fast). Model được nạp
    một lần và tái sử dụng cho mọi lần embed.

    Chạy offline sau khi đã tải trọng số (~2GB); nên bật GPU (BGE_DEVICE=cuda)
    cho throughput index lô lớn. Vector trả về luôn được L2-normalize.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", dim: int = 1024,
                 device: str = "") -> None:
        self.dim = dim
        self.model_name = model_name
        self._backend = ""
        self._model = None
        # 1) FlagEmbedding (native BGE-m3)
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-not-found]

            use_fp16 = device != "cpu"
            kwargs = {"use_fp16": use_fp16}
            if device:
                kwargs["device"] = device
            self._model = BGEM3FlagModel(model_name, **kwargs)
            self._backend = "flagembedding"
            return
        except ImportError:
            pass
        # 2) sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self._model = SentenceTransformer(model_name, device=device or None)
            self._backend = "sentence-transformers"
            return
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "BGEEmbedder cần 'FlagEmbedding' hoặc 'sentence-transformers' "
                "(chưa cài) — cài một trong hai hoặc dùng EMBEDDER=hash"
            ) from exc

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - cần model nặng
        """Vector hóa -> (n, dim) float32, L2-normalized."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._backend == "flagembedding":
            dense = self._model.encode(texts, return_dense=True)["dense_vecs"]
            arr = np.asarray(dense, dtype=np.float32)
        else:  # sentence-transformers
            arr = np.asarray(
                self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
                dtype=np.float32,
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        np.divide(arr, norms, out=arr, where=norms > 0)  # zero-safe, đảm bảo đã chuẩn hóa
        return arr


def get_embedder(settings: Settings) -> Embedder:
    """Factory chọn embedder theo ``settings.embedder``.

    - ``"hash"``   -> ``HashingEmbedder(settings.embedding_dim)`` (offline, fallback)
    - ``"bge"``    -> ``BGEEmbedder(bge_model, bge_dim)`` (BGE-m3 local, chất lượng cao)
    - ``"openai"`` -> ``OpenAIEmbedder(model, api_key)`` (dim mặc định 3072)
    - khác         -> ``ValueError``
    """
    if settings.embedder == "hash":
        return HashingEmbedder(settings.embedding_dim)
    if settings.embedder == "bge":
        return BGEEmbedder(
            model_name=settings.bge_model,
            dim=settings.bge_dim,
            device=settings.bge_device,
        )
    if settings.embedder == "openai":
        return OpenAIEmbedder(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    raise ValueError(f"Embedder không hỗ trợ: {settings.embedder!r}")
