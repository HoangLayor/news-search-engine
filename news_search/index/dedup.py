"""Phát hiện trùng/gần trùng bằng MinHash-LSH thuần Python (F-07).

Mỗi bài được biểu diễn bằng tập shingle (word n-gram trên token đã fold dấu),
nén thành chữ ký MinHash ``num_perm`` giá trị. LSH banding gom ứng viên nhanh,
sau đó xác nhận bằng Jaccard ước lượng >= threshold rồi gán vào cụm sự kiện.

Determinism: hash cơ sở dùng ``zlib.crc32``; các hoán vị (a*h + b) % prime sinh
từ ``random.Random`` với seed cố định — không dùng ``hash()`` built-in.
"""

from __future__ import annotations

import random
import struct
import zlib

from news_search.ingest.tokenizer import fold_tokens, tokenize

# Số nguyên tố Mersenne 2^61 - 1 — đủ lớn so với miền crc32 (2^32).
_PRIME = (1 << 61) - 1
# Seed cố định để chữ ký MinHash deterministic giữa các process.
_SEED = 12345


def _pick_bands(num_perm: int, threshold: float) -> int:
    """Chọn số band b (ước của num_perm) sao cho (1/b)^(1/r) gần threshold nhất."""
    best_b, best_gap = num_perm, float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b:
            continue
        r = num_perm // b
        approx = (1.0 / b) ** (1.0 / r)  # ngưỡng xấp xỉ của đường cong S LSH
        gap = abs(approx - threshold)
        if gap < best_gap:
            best_gap, best_b = gap, b
    return best_b


class MinHashDeduper:
    """Gom cụm bài gần trùng bằng MinHash + LSH banding (CONTRACTS.md §11)."""

    def __init__(self, num_perm: int = 128, threshold: float = 0.7,
                 shingle_size: int = 3) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        # bands * rows == num_perm; bands chọn theo threshold ~ (1/b)^(1/r)
        self.bands = _pick_bands(num_perm, threshold)
        self.rows = num_perm // self.bands
        # Tham số hoán vị (a, b) sinh từ seed cố định -> deterministic.
        rng = random.Random(_SEED)
        self._perms: list[tuple[int, int]] = [
            (rng.randrange(1, _PRIME), rng.randrange(0, _PRIME))
            for _ in range(num_perm)
        ]
        # Cấu trúc trạng thái
        self._signatures: dict[str, tuple[int, ...]] = {}
        # Mỗi band một dict: key band (bytes) -> tập article_id trong bucket.
        self._band_buckets: list[dict[bytes, set[str]]] = [
            {} for _ in range(self.bands)
        ]
        self._cluster_of: dict[str, str] = {}   # article_id -> cluster_id
        self._clusters: dict[str, set[str]] = {}  # cluster_id -> thành viên

    # ------------------------------------------------------------------ #
    # Hỗ trợ nội bộ
    # ------------------------------------------------------------------ #
    def _shingles(self, text: str) -> set[str]:
        """Word n-gram (n=shingle_size) trên fold_tokens(tokenize(text))."""
        tokens = fold_tokens(tokenize(text))
        n = self.shingle_size
        if not tokens:
            return set()
        if len(tokens) < n:
            # Văn bản quá ngắn: dùng cả chuỗi token làm 1 shingle duy nhất.
            return {" ".join(tokens)}
        return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    def _signature(self, shingles: set[str]) -> tuple[int, ...]:
        """Chữ ký MinHash: min của (a*h + b) % prime trên mọi shingle."""
        if not shingles:
            # Không có shingle -> chữ ký sentinel cố định (vẫn deterministic).
            return tuple([_PRIME] * self.num_perm)
        hs = [zlib.crc32(s.encode("utf-8")) for s in shingles]
        return tuple(
            min((a * h + b) % _PRIME for h in hs) for a, b in self._perms
        )

    def _band_keys(self, sig: tuple[int, ...]) -> list[bytes]:
        """Khóa bucket từng band: pack lát cắt chữ ký thành bytes (deterministic)."""
        keys: list[bytes] = []
        for i in range(self.bands):
            chunk = sig[i * self.rows:(i + 1) * self.rows]
            keys.append(struct.pack(f"<{self.rows}Q", *chunk))
        return keys

    @staticmethod
    def _jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
        """Jaccard ước lượng = tỷ lệ vị trí chữ ký trùng nhau."""
        equal = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return equal / len(sig_a)

    # ------------------------------------------------------------------ #
    # API công khai (CONTRACTS.md §11)
    # ------------------------------------------------------------------ #
    def add(self, article_id: str, text: str) -> str:
        """Thêm bài, trả về cluster_id (= id bài đầu tiên của cụm).

        Add lại cùng ``article_id`` -> remove trước rồi add (không trùng lặp).
        """
        if article_id in self._signatures:
            self.remove(article_id)

        sig = self._signature(self._shingles(text))
        keys = self._band_keys(sig)

        # Tìm ứng viên qua LSH: hợp các bucket trùng khóa band.
        candidates: set[str] = set()
        for bucket, key in zip(self._band_buckets, keys):
            candidates |= bucket.get(key, set())
        candidates.discard(article_id)

        # Xác nhận bằng Jaccard ước lượng; chọn ứng viên giống nhất.
        best_id: str | None = None
        best_sim = -1.0
        for cand in sorted(candidates):  # sorted -> deterministic khi hòa điểm
            sim = self._jaccard(sig, self._signatures[cand])
            if sim > best_sim:
                best_id, best_sim = cand, sim

        if best_id is not None and best_sim >= self.threshold:
            cluster_id = self._cluster_of[best_id]
        else:
            cluster_id = article_id  # cụm mới, id cụm = id bài đầu cụm

        # Ghi vào mọi cấu trúc.
        self._signatures[article_id] = sig
        for bucket, key in zip(self._band_buckets, keys):
            bucket.setdefault(key, set()).add(article_id)
        self._cluster_of[article_id] = cluster_id
        self._clusters.setdefault(cluster_id, set()).add(article_id)
        return cluster_id

    def remove(self, article_id: str) -> None:
        """Gỡ bài khỏi mọi cấu trúc; cluster_id của thành viên còn lại GIỮ NGUYÊN."""
        sig = self._signatures.pop(article_id, None)
        if sig is None:
            return  # id không tồn tại -> no-op
        for bucket, key in zip(self._band_buckets, self._band_keys(sig)):
            members = bucket.get(key)
            if members is not None:
                members.discard(article_id)
                if not members:
                    del bucket[key]
        cluster_id = self._cluster_of.pop(article_id)
        cluster = self._clusters.get(cluster_id)
        if cluster is not None:
            cluster.discard(article_id)
            if not cluster:
                del self._clusters[cluster_id]

    def cluster_of(self, article_id: str) -> str | None:
        """cluster_id của bài, None nếu chưa được index."""
        return self._cluster_of.get(article_id)

    def similarity(self, id_a: str, id_b: str) -> float:
        """Jaccard ước lượng giữa hai bài; thiếu id -> 0.0."""
        sig_a = self._signatures.get(id_a)
        sig_b = self._signatures.get(id_b)
        if sig_a is None or sig_b is None:
            return 0.0
        return self._jaccard(sig_a, sig_b)

    def __len__(self) -> int:
        return len(self._signatures)
