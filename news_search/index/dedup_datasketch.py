"""Phát hiện trùng/gần trùng bằng datasketch (MinHash + MinHashLSH) — F-07.

Thay `MinHashDeduper` thuần Python bằng cài đặt tối ưu (nhanh + MinHash chuẩn hơn)
cho quy mô lớn. Cùng interface (CONTRACTS.md §11) nên thay trong suốt qua
``DEDUP_BACKEND=datasketch``. Import-guard: thiếu ``datasketch`` -> RuntimeError.

Determinism: ``datasketch.MinHash`` dùng hoán vị sinh từ ``seed`` cố định.
"""

from __future__ import annotations

from news_search.ingest.tokenizer import fold_tokens, tokenize

_SEED = 12345  # cố định -> chữ ký MinHash deterministic giữa các process


class DatasketchDeduper:
    """Gom cụm bài gần trùng bằng datasketch MinHashLSH (cùng API MinHashDeduper)."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.7,
                 shingle_size: int = 3) -> None:
        try:
            from datasketch import MinHash, MinHashLSH  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "DatasketchDeduper cần package 'datasketch' (chưa cài) — "
                "cài datasketch hoặc dùng DEDUP_BACKEND=local"
            ) from exc
        self._MinHash = MinHash
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._mh: dict = {}                       # article_id -> MinHash
        self._cluster_of: dict[str, str] = {}     # article_id -> cluster_id
        self._clusters: dict[str, set[str]] = {}  # cluster_id -> thành viên

    # ------------------------------------------------------------------ utils
    def _shingles(self, text: str) -> set[str]:
        """Word n-gram (n=shingle_size) trên fold_tokens(tokenize(text))."""
        tokens = fold_tokens(tokenize(text))
        n = self.shingle_size
        if not tokens:
            return set()
        if len(tokens) < n:
            return {" ".join(tokens)}
        return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    def _make(self, shingles: set[str]):
        m = self._MinHash(num_perm=self.num_perm, seed=_SEED)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    # -------------------------------------------------------------------- API
    def add(self, article_id: str, text: str) -> str:
        if article_id in self._mh:
            self.remove(article_id)
        m = self._make(self._shingles(text))
        candidates = [c for c in self._lsh.query(m) if c != article_id]

        best_id, best_sim = None, -1.0
        for cand in sorted(candidates):  # sorted -> deterministic khi hòa
            sim = m.jaccard(self._mh[cand])
            if sim > best_sim:
                best_id, best_sim = cand, sim

        cluster_id = (
            self._cluster_of[best_id]
            if best_id is not None and best_sim >= self.threshold
            else article_id
        )
        self._mh[article_id] = m
        self._lsh.insert(article_id, m)
        self._cluster_of[article_id] = cluster_id
        self._clusters.setdefault(cluster_id, set()).add(article_id)
        return cluster_id

    def remove(self, article_id: str) -> None:
        if article_id not in self._mh:
            return  # no-op
        del self._mh[article_id]
        try:
            self._lsh.remove(article_id)
        except Exception:  # phiên bản/khóa không có -> bỏ qua
            pass
        cluster_id = self._cluster_of.pop(article_id)
        cluster = self._clusters.get(cluster_id)
        if cluster is not None:
            cluster.discard(article_id)
            if not cluster:
                del self._clusters[cluster_id]

    def cluster_of(self, article_id: str) -> str | None:
        return self._cluster_of.get(article_id)

    def similarity(self, id_a: str, id_b: str) -> float:
        ma, mb = self._mh.get(id_a), self._mh.get(id_b)
        if ma is None or mb is None:
            return 0.0
        return float(ma.jaccard(mb))

    def __len__(self) -> int:
        return len(self._mh)
