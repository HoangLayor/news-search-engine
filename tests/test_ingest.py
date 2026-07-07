"""Test module INGEST (F-02, F-03): tokenizer + cleaner.

Deterministic: datetime cố định, không phụ thuộc giờ thực.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from news_search.ingest import tokenizer
from news_search.ingest.cleaner import TZ_VN, clean_html, normalize_article
from news_search.ingest.tokenizer import (
    fold_diacritics,
    fold_tokens,
    normalize_text,
    tokenize,
)
from news_search.models import Article


# ---------------------------------------------------------------- tokenizer

class TestNormalizeText:
    def test_lowercase_va_gon_khoang_trang(self):
        assert normalize_text("  Giá  Vàng \t Hôm\nNay  ") == "giá vàng hôm nay"

    def test_chuoi_rong(self):
        assert normalize_text("   ") == ""


class TestFoldDiacritics:
    def test_bo_dau_dong_thap(self):
        # Yêu cầu đề bài: "Đồng Tháp" -> "Dong Thap" (giữ hoa/thường)
        assert fold_diacritics("Đồng Tháp") == "Dong Thap"

    def test_bo_dau_du_ky_tu(self):
        assert fold_diacritics("đường ướt ễnh ộp") == "duong uot enh op"

    def test_khong_lowercase(self):
        assert fold_diacritics("VIỆT Nam") == "VIET Nam"

    def test_chuoi_khong_dau_giu_nguyen(self):
        assert fold_diacritics("hello 123") == "hello 123"


class TestTokenize:
    def test_tieng_viet_co_dau_fallback(self, monkeypatch):
        # Ép dùng fallback regex để test deterministic mọi môi trường
        monkeypatch.setattr(tokenizer, "_underthesea_word_tokenize", None)
        assert tokenize("Giá vàng SJC, hôm nay tăng 2%!") == [
            "giá", "vàng", "sjc", "hôm", "nay", "tăng", "2",
        ]

    def test_giu_dau_va_lowercase(self):
        # Bất kể backend nào: token phải lowercase, GIỮ dấu, không dấu câu
        toks = tokenize("Đồng Tháp đón mưa lớn.")
        joined = " ".join(toks).replace("_", " ")
        assert "đồng" in joined and "tháp" in joined
        for tok in toks:
            assert tok == tok.lower()
            assert "." not in tok and "," not in tok

    def test_chuoi_rong(self):
        assert tokenize("   ") == []


class TestFoldTokens:
    def test_fold_tung_token(self):
        assert fold_tokens(["đồng", "tháp", "việt"]) == ["dong", "thap", "viet"]

    def test_list_rong(self):
        assert fold_tokens([]) == []


# ------------------------------------------------------------------ cleaner

class TestCleanHtml:
    def test_bo_script_style_giu_doan(self):
        html = (
            "<html><head><style>p{color:red}</style>"
            "<script>var x = 1;</script></head>"
            "<body><p>Đoạn một.</p><p>Đoạn hai &amp; ba.</p>"
            "<div>Cuối<br>dòng</div></body></html>"
        )
        out = clean_html(html)
        assert "var x" not in out
        assert "color" not in out
        assert "Đoạn một.\nĐoạn hai & ba." in out  # <p> -> ngắt đoạn
        assert "Cuối\ndòng" in out  # <br> -> ngắt đoạn
        assert "<" not in out and ">" not in out

    def test_unescape_entity(self):
        assert clean_html("A &lt;b&gt; &quot;C&quot; &nbsp; D") == 'A <b> "C" D'

    def test_gon_khoang_trang_trong_dong(self):
        assert clean_html("<p>  nhiều   khoảng \t trắng </p>") == "nhiều khoảng trắng"

    def test_chuoi_rong(self):
        assert clean_html("") == ""


class TestNormalizeArticle:
    def _raw(self, **override) -> dict:
        raw = {
            "article_id": "a1",
            "title": "  Tin nóng Đồng Tháp  ",
            "body": "Nội dung bài viết.",
            "url": "https://example.com/a1",
            "published_at": "2026-07-01T10:00:00+07:00",
        }
        raw.update(override)
        return raw

    def test_du_field(self):
        art = normalize_article(
            self._raw(author=" Nguyễn Văn A ", category="Thời sự", source=" VnExpress ")
        )
        assert isinstance(art, Article)
        assert art.article_id == "a1"
        assert art.title == "Tin nóng Đồng Tháp"  # đã strip
        assert art.author == "Nguyễn Văn A"
        assert art.category == "Thời sự"
        assert art.source == "VnExpress"
        assert art.status == "published"  # mặc định
        assert art.tags == []  # mặc định

    def test_thieu_field_bat_buoc(self):
        raw = self._raw()
        del raw["title"]
        with pytest.raises(ValueError, match="title"):
            normalize_article(raw)

    def test_field_bat_buoc_rong(self):
        with pytest.raises(ValueError, match="url"):
            normalize_article(self._raw(url=""))

    def test_published_at_chuoi_iso_co_timezone(self):
        art = normalize_article(self._raw(published_at="2026-07-01T10:00:00+00:00"))
        assert art.published_at == datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
        assert art.published_at.utcoffset() == timedelta(0)

    def test_published_at_chuoi_iso_khong_timezone(self):
        # Chuỗi naive -> gán UTC+7
        art = normalize_article(self._raw(published_at="2026-07-01T10:00:00"))
        assert art.published_at.tzinfo is not None
        assert art.published_at.utcoffset() == timedelta(hours=7)
        assert art.published_at == datetime(2026, 7, 1, 10, 0, tzinfo=TZ_VN)

    def test_published_at_datetime_naive(self):
        art = normalize_article(self._raw(published_at=datetime(2026, 7, 1, 8, 30)))
        assert art.published_at == datetime(2026, 7, 1, 8, 30, tzinfo=TZ_VN)

    def test_published_at_datetime_aware_giu_nguyen(self):
        dt = datetime(2026, 7, 1, 3, 0, tzinfo=timezone.utc)
        art = normalize_article(self._raw(published_at=dt))
        assert art.published_at is dt

    def test_published_at_khong_hop_le(self):
        with pytest.raises(ValueError, match="published_at"):
            normalize_article(self._raw(published_at="hôm qua"))

    def test_body_html_duoc_lam_sach(self):
        art = normalize_article(
            self._raw(
                body="<p>Đoạn một.</p><script>alert(1)</script><p>Đoạn hai.</p>"
            )
        )
        assert art.body == "Đoạn một.\nĐoạn hai."
        assert "alert" not in art.body

    def test_body_khong_html_giu_nguyen(self):
        art = normalize_article(self._raw(body="Giá 1 < 2 và 3 > 2 nhé."))
        assert art.body == "Giá 1 < 2 và 3 > 2 nhé."

    def test_tags_va_status(self):
        art = normalize_article(
            self._raw(tags=[" kinh tế ", "vàng", ""], status="unpublished")
        )
        assert art.tags == ["kinh tế", "vàng"]
        assert art.status == "unpublished"
