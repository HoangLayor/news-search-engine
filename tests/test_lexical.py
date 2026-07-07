"""Test LexicalIndex (BM25, F-04/F-09) và make_snippet (F-15).

Nếu ``news_search.ingest.tokenizer`` (agent khác viết song song) chưa tồn tại,
tạo stub tối giản đúng contract vào sys.modules TRƯỚC khi import lexical —
test độc lập nhưng production vẫn dùng tokenizer thật khi có.
Mọi datetime cố định -> deterministic, không phụ thuộc giờ thực.
"""

from __future__ import annotations

import re
import sys
import types
import unicodedata
from datetime import datetime, timezone


def _install_tokenizer_stub() -> None:
    """Tạo stub tokenizer đúng contract §1 (fold bằng unicodedata, tokenize \\w+)."""
    mod = types.ModuleType("news_search.ingest.tokenizer")

    def normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFC", text).lower()
        return " ".join(text.split())

    def fold_diacritics(text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return stripped.replace("đ", "d").replace("Đ", "d")

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", normalize_text(text), re.UNICODE)

    def fold_tokens(tokens: list[str]) -> list[str]:
        return [fold_diacritics(t) for t in tokens]

    mod.normalize_text = normalize_text
    mod.fold_diacritics = fold_diacritics
    mod.tokenize = tokenize
    mod.fold_tokens = fold_tokens
    sys.modules["news_search.ingest.tokenizer"] = mod
    import news_search.ingest as ingest_pkg

    ingest_pkg.tokenizer = mod  # để `from news_search.ingest import tokenizer` cũng chạy


try:  # tokenizer thật (nếu agent kia đã viết) — ưu tiên dùng
    import news_search.ingest.tokenizer  # noqa: F401
except ImportError:
    _install_tokenizer_stub()

from news_search.index.lexical import LexicalIndex  # noqa: E402
from news_search.models import Article  # noqa: E402
from news_search.search.snippet import make_snippet  # noqa: E402

# Thời điểm cố định cho mọi bài viết (timezone-aware, deterministic)
NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _article(article_id: str, title: str, body: str) -> Article:
    return Article(
        article_id=article_id,
        title=title,
        body=body,
        url=f"https://news.example/{article_id}",
        published_at=NOW,
    )


# ----------------------------------------------------------------- BM25 core


def test_bm25_doc_nhieu_term_khop_xep_cao_hon():
    """Doc chứa cả 2 term truy vấn phải đứng trên doc chỉ chứa 1 term."""
    idx = LexicalIndex()
    idx.add(_article("d1", "tin tức buổi sáng", "lạm phát tăng khiến ngân hàng điều chỉnh lãi suất"))
    idx.add(_article("d2", "tin tức buổi chiều", "lạm phát tăng nhẹ trong quý một năm nay rồi"))
    idx.add(_article("d3", "tin tức buổi tối", "bóng đá trong nước có nhiều trận đấu hay"))

    results = idx.search("lạm phát ngân hàng")
    ids = [doc_id for doc_id, _ in results]
    assert ids[0] == "d1"  # khớp cả "lạm phát" lẫn "ngân hàng"
    assert "d2" in ids and ids.index("d1") < ids.index("d2")
    assert "d3" not in ids  # không khớp term nào
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)  # giảm dần
    assert all(score > 0 for score in scores)


def test_match_khong_dau():
    """Query không dấu "lam phat" phải tìm ra bài viết "lạm phát"."""
    idx = LexicalIndex()
    idx.add(_article("d1", "kinh tế vĩ mô", "chỉ số lạm phát tháng này tăng cao kỷ lục"))
    idx.add(_article("d2", "thể thao", "đội tuyển quốc gia thắng trận giao hữu"))

    results = idx.search("lam phat")
    assert [doc_id for doc_id, _ in results] == ["d1"]
    assert results[0][1] > 0
    # Chiều ngược lại: query có dấu vẫn khớp (index folded)
    assert [doc_id for doc_id, _ in idx.search("lạm phát")] == ["d1"]


def test_title_weight_uu_tien_khop_tieu_de():
    """Cùng term: khớp ở tiêu đề (x3) phải điểm cao hơn khớp ở body."""
    idx = LexicalIndex()
    # Tiêu đề cùng 4 token, body cùng 6 token -> độ dài doc bằng nhau
    idx.add(_article("t1", "kinh tế việt nam", "nội dung nói về tăng trưởng chung"))
    idx.add(_article("t2", "bản tin buổi sáng", "kinh tế được nhắc tới tại đây"))

    results = dict(idx.search("kinh tế"))
    assert results["t1"] > results["t2"]


def test_allowed_ids_loc_dung():
    """allowed_ids != None -> chỉ chấm điểm doc trong tập đó."""
    idx = LexicalIndex()
    idx.add(_article("d1", "lạm phát", "lạm phát ảnh hưởng giá cả"))
    idx.add(_article("d2", "lạm phát quý hai", "lạm phát tiếp tục tăng"))

    results = idx.search("lạm phát", allowed_ids={"d2"})
    assert [doc_id for doc_id, _ in results] == ["d2"]
    assert idx.search("lạm phát", allowed_ids=set()) == []
    # None -> không lọc
    assert {doc_id for doc_id, _ in idx.search("lạm phát")} == {"d1", "d2"}


def test_remove_va_re_add_khong_trung():
    """remove gỡ sạch; re-add cùng id = replace, không nhân đôi."""
    idx = LexicalIndex()
    idx.add(_article("d1", "lạm phát", "lạm phát tăng"))
    idx.add(_article("d2", "thể thao", "bóng đá hôm nay"))
    assert len(idx) == 2 and "d1" in idx

    idx.remove("d1")
    assert len(idx) == 1 and "d1" not in idx
    assert idx.search("lạm phát") == []
    idx.remove("khong-ton-tai")  # no-op, không raise
    assert len(idx) == 1

    # Re-add cùng id 2 lần -> vẫn chỉ 1 doc (d2), xuất hiện đúng 1 lần trong kết quả
    idx.add(_article("d2", "thể thao mới", "bóng đá hôm nay thắng lớn"))
    idx.add(_article("d2", "kinh tế thay thế", "giá vàng biến động mạnh"))
    assert len(idx) == 1
    assert idx.search("bóng đá") == []  # nội dung cũ đã bị thay
    hits = idx.search("giá vàng")
    assert [doc_id for doc_id, _ in hits].count("d2") == 1


def test_update_va_query_rong():
    """update = remove + add; query rỗng -> []."""
    idx = LexicalIndex()
    idx.add(_article("d1", "tin cũ", "nội dung cũ về chứng khoán"))
    idx.update(_article("d1", "tin mới", "nội dung mới về bất động sản"))
    assert len(idx) == 1
    assert idx.search("chứng khoán") == []
    assert [doc_id for doc_id, _ in idx.search("bất động sản")] == ["d1"]
    assert idx.search("") == []
    assert idx.search("   ") == []


# ----------------------------------------------------------------- Snippet


def test_snippet_boi_dam_tu_co_dau_khi_query_khong_dau():
    """Query không dấu -> bôi đậm đúng chữ gốc CÓ DẤU trong body."""
    body = "Chỉ số lạm phát tháng sáu tăng mạnh, gây áp lực lên lãi suất ngân hàng."
    snippet = make_snippet(body, ["lam", "phat"], max_len=200)
    assert "<b>lạm</b>" in snippet
    assert "<b>phát</b>" in snippet
    assert "lam" not in snippet.replace("<b>", "").replace("</b>", "")  # giữ nguyên dấu


def test_snippet_do_dai_va_ellipsis():
    """Phần chữ (bỏ tag <b>) <= max_len + ellipsis hai đầu; có '...' khi cắt."""
    body = ("mở đầu dài dòng không liên quan " * 10
            + "đoạn giữa nói về lạm phát và ngân hàng rất chi tiết "
            + "phần kết cũng dài dòng không kém " * 10)
    max_len = 80
    snippet = make_snippet(body, ["lam phat", "ngan hang"], max_len=max_len)
    visible = snippet.replace("<b>", "").replace("</b>", "")
    assert len(visible) <= max_len + 2 * len("...")
    assert snippet.startswith("...") and snippet.endswith("...")  # cắt cả 2 đầu
    assert "<b>lạm</b>" in snippet or "<b>phát</b>" in snippet


def test_snippet_chon_cua_so_nhieu_match_nhat():
    """Cửa sổ phải phủ cụm chứa NHIỀU match nhất, không phải match đầu tiên."""
    body = ("giá vàng biến động nhẹ hôm qua. " + "chữ đệm " * 30
            + "giá vàng và giá xăng cùng tăng, giá điện giữ nguyên. "
            + "chữ đệm " * 30)
    snippet = make_snippet(body, ["gia"], max_len=60)
    # Cụm sau có 3 lần "giá" -> phải nằm trong snippet
    assert snippet.count("<b>giá</b>") >= 2


def test_snippet_khong_match_va_body_rong():
    """Không match -> đầu body cắt max_len (+'...'); body rỗng -> ''."""
    body = "Bài viết nói về thể thao trong nước và quốc tế hôm nay."
    snippet = make_snippet(body, ["blockchain"], max_len=20)
    assert snippet == body[:20] + "..."
    assert "<b>" not in snippet
    # Body ngắn hơn max_len -> không ellipsis
    assert make_snippet("Ngắn thôi.", ["blockchain"], max_len=50) == "Ngắn thôi."
    assert make_snippet("", ["lam"], max_len=50) == ""


def test_snippet_match_tu_nguyen_ven():
    """Chỉ match từ nguyên vẹn: token 'gia' không được bôi đậm trong 'giao'."""
    body = "Gia đình tham gia giao thông; giá cả tăng."
    snippet = make_snippet(body, ["gia"], max_len=200)
    # "gia" khớp từ nguyên vẹn: "Gia", "gia", "giá" (không dấu/hoa thường);
    # "giao" là từ khác -> tuyệt đối không được khớp.
    assert "<b>giao</b>" not in snippet
    assert "<b>Gia</b>" in snippet  # giữ nguyên chữ hoa gốc
    assert "<b>giá</b>" in snippet
