"""Test MinHashDeduper (F-07) và collapse_clusters/mmr (F-14).

Dùng tokenizer thật (news_search.ingest.tokenizer đã có trên đĩa). Mọi cấu trúc
deterministic (seed cố định trong dedup) nên kết quả ổn định giữa các lần chạy.
"""

from __future__ import annotations

import numpy as np

from news_search.index.dedup import MinHashDeduper
from news_search.search.diversify import collapse_clusters, mmr

# Hai bản tin GẦN TRÙNG: chỉ khác đúng 1 từ ("sáng" -> "chiều").
_NEAR_A = (
    "ủy ban bầu cử quốc gia công bố kết quả chính thức của cuộc bầu cử tổng "
    "thống diễn ra vào sáng nay tại thủ đô sau nhiều giờ kiểm phiếu căng thẳng"
)
_NEAR_B = (
    "ủy ban bầu cử quốc gia công bố kết quả chính thức của cuộc bầu cử tổng "
    "thống diễn ra vào chiều nay tại thủ đô sau nhiều giờ kiểm phiếu căng thẳng"
)
# Bản tin KHÁC HẲN chủ đề.
_DIFFERENT = (
    "đội tuyển bóng đá quốc gia giành chiến thắng thuyết phục trong trận chung "
    "kết giải vô địch khu vực trước sự cổ vũ của hàng vạn khán giả nhà"
)


# ----------------------------------------------------------------- MinHash


def test_near_duplicate_cung_cluster():
    """Hai bài gần trùng vào CÙNG cluster; bài khác hẳn ra cluster riêng."""
    d = MinHashDeduper(num_perm=128, threshold=0.6, shingle_size=3)
    c_a = d.add("a", _NEAR_A)
    c_b = d.add("b", _NEAR_B)
    c_c = d.add("c", _DIFFERENT)

    assert c_a == c_b  # gần trùng -> chung cụm
    assert d.cluster_of("a") == d.cluster_of("b")
    assert c_c != c_a  # khác hẳn -> cụm riêng
    assert d.cluster_of("c") == "c"  # cluster_id = id bài đầu cụm


def test_similarity_cao_cho_cap_gan_trung():
    """similarity() cao cho cặp gần trùng, thấp cho cặp khác chủ đề; thiếu id -> 0."""
    d = MinHashDeduper(num_perm=128, threshold=0.6, shingle_size=3)
    d.add("a", _NEAR_A)
    d.add("b", _NEAR_B)
    d.add("c", _DIFFERENT)

    assert d.similarity("a", "b") >= 0.6
    assert d.similarity("a", "c") < 0.3
    assert d.similarity("a", "khong-ton-tai") == 0.0


def test_remove_giu_cluster_id_thanh_vien_con_lai():
    """Gỡ 1 bài khỏi cụm 3 bài -> 2 bài còn lại GIỮ NGUYÊN cluster_id."""
    d = MinHashDeduper(num_perm=128, threshold=0.6, shingle_size=3)
    d.add("a", _NEAR_A)
    d.add("b", _NEAR_B)
    # Bài thứ 3 cũng gần trùng (khác 1 từ khác)
    near_c = _NEAR_A.replace("thủ đô", "trung tâm")
    d.add("c", near_c)
    cluster = d.cluster_of("a")
    assert d.cluster_of("b") == cluster and d.cluster_of("c") == cluster

    d.remove("a")
    assert d.cluster_of("a") is None  # đã gỡ
    assert d.cluster_of("b") == cluster  # thành viên còn lại giữ nguyên
    assert d.cluster_of("c") == cluster
    assert len(d) == 2


def test_add_cung_id_khong_phinh():
    """Add lại cùng article_id = replace, không nhân đôi."""
    d = MinHashDeduper(num_perm=64, threshold=0.7, shingle_size=3)
    d.add("x", _NEAR_A)
    d.add("x", _DIFFERENT)  # thay nội dung
    assert len(d) == 1
    # Nội dung mới -> giống _DIFFERENT hơn
    d.add("y", _DIFFERENT)
    assert d.similarity("x", "y") >= 0.6
    d.remove("khong-ton-tai")  # no-op
    assert len(d) == 2


# --------------------------------------------------------------- Diversify


def test_collapse_clusters_giu_mot_bai_moi_cum():
    """Mỗi cụm giữ tối đa max_per_cluster bài, đúng thứ tự ranking."""
    ranked = ["a", "b", "c", "d", "e"]
    cluster = {"a": "C1", "b": "C1", "c": "C2", "d": None, "e": "C2"}
    kept = collapse_clusters(ranked, lambda i: cluster.get(i), max_per_cluster=1)
    assert kept == ["a", "c", "d"]  # b (trùng C1), e (trùng C2) bị loại

    kept2 = collapse_clusters(ranked, lambda i: cluster.get(i), max_per_cluster=2)
    assert kept2 == ["a", "b", "c", "d", "e"]  # cho phép 2 bài/cụm


def test_mmr_da_dang_hoa():
    """MMR ưu tiên đa dạng: bài thứ 2 phải là bài KHÁC HƯỚNG, không phải bản trùng."""
    vectors = {
        "i0": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "i1": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # khác hướng i0
        "i2": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # trùng hướng i0
    }
    ranked = [("i0", 1.0), ("i1", 0.9), ("i2", 0.8)]
    order = mmr(ranked, lambda i: vectors.get(i), lambda_=0.7, top_k=3)
    assert order[0] == "i0"  # điểm cao nhất chọn trước
    assert order[1] == "i1"  # đa dạng: chọn bài khác hướng thay vì bản gần trùng i2
    assert order[2] == "i2"


def test_mmr_vector_lookup_none():
    """vector_lookup trả None -> vẫn xếp theo độ liên quan, không lỗi."""
    ranked = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    order = mmr(ranked, lambda i: None, lambda_=0.7, top_k=2)
    assert order == ["a", "b"]  # sim = 0 với mọi doc -> thuần relevance
    assert mmr([], lambda i: None, top_k=5) == []
