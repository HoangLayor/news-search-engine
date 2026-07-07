"""Test extract_entities + KnowledgeGraph (F-06).

Fallback heuristic deterministic (không cần underthesea). Test khoan dung với
ranh giới cụm nhưng PHẢI khẳng định được các thực thể chính.
"""

from __future__ import annotations

from news_search.index.entities import KnowledgeGraph, extract_entities
from news_search.models import Entity

_SENT = (
    "Thủ tướng Phạm Minh Chính làm việc với UBND tỉnh Đồng Tháp "
    "và Tập đoàn Vingroup."
)


def test_extract_entities_cau_tieng_viet():
    """Trích được PER người, ORG tổ chức từ câu tiếng Việt thật."""
    ents = extract_entities(_SENT)
    names = [e.name for e in ents]

    # PER: tên người 3 âm tiết
    assert any(e.type == "PER" and "Phạm Minh Chính" in e.name for e in ents), names
    # ORG: ít nhất một tổ chức chứa "Vingroup" hoặc "Đồng Tháp"
    assert any(
        e.type == "ORG" and ("Vingroup" in e.name or "Đồng Tháp" in e.name)
        for e in ents
    ), names
    # "Thủ" (stopword đầu câu) không được đứng riêng thành thực thể
    assert "Thủ" not in names


def test_extract_entities_rong_va_dedup():
    """Text rỗng -> []; thực thể lặp lại chỉ xuất hiện một lần."""
    assert extract_entities("") == []
    assert extract_entities("   ") == []
    ents = extract_entities(
        "Ông Nguyễn Văn A phát biểu. Sau đó Nguyễn Văn A rời đi."
    )
    per = [e for e in ents if e.type == "PER" and "Nguyễn Văn A" in e.name]
    assert len(per) == 1  # dedup theo (name, type)


def test_loc_nhieu_misc_mot_tu():
    """Bỏ MISC một-từ title-case (từ đầu câu bị nhận nhầm), giữ acronym viết hoa."""
    # "Phân", "Giới" là từ đầu câu -> không được thành thực thể
    ents = extract_entities("Phân tích cho thấy xu hướng mới. Giới quan sát đồng tình.")
    assert all(e.name not in ("Phân", "Giới") for e in ents)
    # Acronym viết hoa toàn bộ vẫn được giữ
    unesco = extract_entities("Di sản được UNESCO công nhận")
    assert any(e.name == "UNESCO" for e in unesco)


def test_knowledge_graph_add_remove_query():
    """KnowledgeGraph: add/remove + tra cứu theo tên không dấu."""
    kg = KnowledgeGraph()
    kg.add_article("art-1", extract_entities(_SENT))
    kg.add_article(
        "art-2",
        [Entity("Phạm Minh Chính", "PER"), Entity("Hà Nội", "LOC")],
    )

    # Tra theo tên không dấu, không phân biệt hoa/thường
    assert kg.articles_for_entity("pham minh chinh") == {"art-1", "art-2"}
    assert kg.articles_for_entity("PHẠM MINH CHÍNH") == {"art-1", "art-2"}
    assert kg.articles_for_entity("ha noi") == {"art-2"}
    assert kg.articles_for_entity("không tồn tại") == set()

    # entities_for_article trả đúng danh sách
    assert any(e.name == "Hà Nội" for e in kg.entities_for_article("art-2"))

    # remove gỡ bài; thực thể mồ côi biến mất, thực thể chung vẫn còn
    kg.remove_article("art-2")
    assert kg.articles_for_entity("ha noi") == set()
    assert kg.articles_for_entity("pham minh chinh") == {"art-1"}


def test_knowledge_graph_top_entities():
    """top_entities đếm đúng số bài chứa mỗi thực thể, giảm dần."""
    kg = KnowledgeGraph()
    common = Entity("Chính Phủ", "ORG")
    kg.add_article("a", [common, Entity("Hà Nội", "LOC")])
    kg.add_article("b", [common, Entity("Đà Nẵng", "LOC")])
    kg.add_article("c", [common])

    top = kg.top_entities(n=3)
    assert top[0][0].name == "Chính Phủ" and top[0][1] == 3  # xuất hiện 3 bài
    counts = {ent.name: n for ent, n in top}
    assert counts.get("Hà Nội") == 1
