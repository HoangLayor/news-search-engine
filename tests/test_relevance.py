"""Test module RELEVANCE (F-11, F-13): parse_query, rrf, time-decay, rerank.

Deterministic: dùng ``NOW`` cố định, không phụ thuộc giờ thực.
Nếu ``news_search.ingest.tokenizer`` chưa tồn tại (module của agent khác),
stub vào ``sys.modules`` theo mô tả ở CONTRACTS.md §7.
"""

import re
import sys
import types
import unicodedata
from datetime import datetime, timedelta, timezone

import pytest


def _install_tokenizer_stub() -> None:
    """Stub tokenizer (fold bằng unicodedata, tokenize regex) nếu module thật chưa có."""
    try:
        import news_search.ingest.tokenizer  # noqa: F401

        return
    except ImportError:
        pass

    stub = types.ModuleType("news_search.ingest.tokenizer")

    def normalize_text(text: str) -> str:
        # NFC, lowercase, gọn khoảng trắng
        text = unicodedata.normalize("NFC", text).lower()
        return re.sub(r"\s+", " ", text).strip()

    def fold_diacritics(text: str) -> str:
        # NFD + bỏ combining marks; đ/Đ -> d
        decomposed = unicodedata.normalize("NFD", text)
        stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return stripped.replace("đ", "d").replace("Đ", "d")

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", normalize_text(text), re.UNICODE)

    def fold_tokens(tokens: list[str]) -> list[str]:
        return [fold_diacritics(tok) for tok in tokens]

    stub.normalize_text = normalize_text
    stub.fold_diacritics = fold_diacritics
    stub.tokenize = tokenize
    stub.fold_tokens = fold_tokens
    sys.modules["news_search.ingest.tokenizer"] = stub


_install_tokenizer_stub()

from news_search.config import Settings  # noqa: E402
from news_search.search.fusion import rrf  # noqa: E402
from news_search.search.query import FRESH_HINT_TOKENS, parse_query  # noqa: E402
from news_search.search.ranking import apply_time_decay, time_decay_multiplier  # noqa: E402
from news_search.search.rerank import NoopReranker, get_reranker  # noqa: E402

# Thời điểm "hiện tại" cố định cho mọi test time-decay
NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- parse_query
class TestParseQuery:
    def test_fresh_intent_cum_hom_nay(self):
        parsed = parse_query("động đất hôm nay")
        assert parsed.fresh_intent is True
        assert parsed.folded == "dong dat hom nay"

    def test_fresh_intent_moi_nhat(self):
        assert parse_query("tin mới nhất").fresh_intent is True

    def test_khong_fresh_lam_phat_thang_6(self):
        assert parse_query("lạm phát tháng 6").fresh_intent is False

    def test_moi_khong_an_nham_moi_truong(self):
        # "môi trường" fold thành "moi truong" — không được coi là hint "mới"
        assert parse_query("ô nhiễm môi trường").fresh_intent is False

    def test_moi_dung_rieng_van_fresh(self):
        # "mới" đứng riêng (không đi với "trường") vẫn là hint tin mới
        assert parse_query("tin mới về môi trường").fresh_intent is True

    def test_query_rong_raise(self):
        with pytest.raises(ValueError):
            parse_query("")

    def test_query_toan_whitespace_raise(self):
        with pytest.raises(ValueError):
            parse_query("   \t\n  ")

    def test_cac_truong_parsed_query(self):
        raw = "  Động   Đất "
        parsed = parse_query(raw)
        assert parsed.raw == raw
        assert parsed.normalized == "động đất"
        assert parsed.folded == "dong dat"
        assert parsed.tokens == ["động", "đất"]
        assert parsed.fresh_intent is False

    def test_fresh_hint_tokens_da_folded(self):
        # Hằng số phải ở dạng folded và chứa tối thiểu các hint chính
        assert {"hom nay", "moi nhat", "moi", "breaking", "live", "truc tiep"} <= set(
            FRESH_HINT_TOKENS
        )
        for hint in FRESH_HINT_TOKENS:
            assert hint == hint.lower()


# ------------------------------------------------------------------------ rrf
class TestRRF:
    def test_cong_thuc_1_tren_k_cong_rank(self):
        scores = rrf([["a", "b", "c"]], k=60)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)
        assert scores["c"] == pytest.approx(1 / 63)

    def test_cong_don_giua_cac_ranking(self):
        scores = rrf([["a", "b", "c"], ["b", "d"]], k=10)
        assert scores["a"] == pytest.approx(1 / 11)
        assert scores["b"] == pytest.approx(1 / 12 + 1 / 11)  # rank 2 + rank 1
        assert scores["c"] == pytest.approx(1 / 13)
        assert scores["d"] == pytest.approx(1 / 12)

    def test_ranking_rong_bo_qua(self):
        scores = rrf([[], ["x"]], k=60)
        assert scores == {"x": pytest.approx(1 / 61)}

    def test_khong_ranking_nao(self):
        assert rrf([]) == {}


# ----------------------------------------------------------------- time-decay
class TestTimeDecay:
    def test_bai_dung_half_life(self):
        published = NOW - timedelta(days=7)
        m = time_decay_multiplier(published, NOW, half_life_days=7.0, floor=0.3)
        assert m == pytest.approx(0.3 + (1 - 0.3) * 0.5)

    def test_bai_tuoi_0_multiplier_1(self):
        assert time_decay_multiplier(NOW, NOW, half_life_days=7.0) == pytest.approx(1.0)

    def test_bai_tuong_lai_kep_ve_1(self):
        future = NOW + timedelta(days=3)
        assert time_decay_multiplier(future, NOW, half_life_days=7.0) == pytest.approx(1.0)

    def test_half_life_khong_duong_tat_decay(self):
        old = NOW - timedelta(days=100)
        assert time_decay_multiplier(old, NOW, half_life_days=0.0) == 1.0
        assert time_decay_multiplier(old, NOW, half_life_days=-2.5) == 1.0

    def test_apply_time_decay(self):
        published = {"new": NOW, "old": NOW - timedelta(days=7)}
        scores = {"new": 2.0, "old": 2.0}
        result = apply_time_decay(
            scores, published.__getitem__, NOW, half_life_days=7.0, floor=0.3
        )
        assert result["new"] == pytest.approx(2.0)
        assert result["old"] == pytest.approx(2.0 * 0.65)
        # trả dict mới, không sửa dict vào
        assert scores == {"new": 2.0, "old": 2.0}


# --------------------------------------------------------------------- rerank
class TestRerank:
    DOCS = [("a", "văn bản a"), ("b", "văn bản b"), ("c", "văn bản c")]

    def test_noop_giu_thu_tu_score_1_tren_rank(self):
        out = NoopReranker().rerank("truy vấn", self.DOCS)
        assert [doc_id for doc_id, _ in out] == ["a", "b", "c"]
        assert [s for _, s in out] == [
            pytest.approx(1.0),
            pytest.approx(1 / 2),
            pytest.approx(1 / 3),
        ]

    def test_noop_top_k(self):
        out = NoopReranker().rerank("q", self.DOCS, top_k=2)
        assert [doc_id for doc_id, _ in out] == ["a", "b"]

    def test_noop_docs_rong(self):
        assert NoopReranker().rerank("q", []) == []

    def test_get_reranker_none(self):
        reranker = get_reranker(Settings(reranker="none"))
        assert isinstance(reranker, NoopReranker)

    def test_get_reranker_gia_tri_la_raise(self):
        with pytest.raises(ValueError):
            get_reranker(Settings(reranker="sieu-ai"))
