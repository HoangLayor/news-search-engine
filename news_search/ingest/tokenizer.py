"""Chuẩn hóa & token hóa văn bản tiếng Việt (F-03).

Bộ tách từ chọn theo cấu hình ``TOKENIZER`` (mặc định ``auto``):
    auto  -> pyvi -> underthesea -> regex (dùng cái nào cài được, ưu tiên trái)
    pyvi  -> ép pyvi (tách từ ghép "bất_động_sản" -> BM25 tiếng Việt chính xác hơn)
    underthesea | regex -> ép tương ứng
pyvi/underthesea tách **từ ghép** (nối bằng "_"), nâng độ chính xác lexical rõ rệt
so với regex tách theo âm tiết. Mọi hàm deterministic, không I/O.
"""

from __future__ import annotations

import re
import unicodedata

from news_search.config import Settings

# Gom mọi whitespace (space, tab, newline, nbsp...) về 1 space
_WS_RE = re.compile(r"\s+")
# Token = chuỗi chữ/số/underscore (unicode); dấu câu bị loại
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# --- Chọn bộ tách từ MỘT LẦN lúc import (đọc mode qua config -> .env đã nạp) ---
_MODE = Settings().tokenizer.strip().lower()
_pyvi_tokenize = None
_underthesea_word_tokenize = None

if _MODE in ("auto", "pyvi"):
    try:  # pragma: no cover - phụ thuộc môi trường
        from pyvi import ViTokenizer

        _pyvi_tokenize = ViTokenizer.tokenize
    except ImportError:  # pragma: no cover
        _pyvi_tokenize = None

if _pyvi_tokenize is None and _MODE in ("auto", "underthesea"):
    try:  # pragma: no cover
        from underthesea import word_tokenize as _underthesea_word_tokenize
    except ImportError:  # pragma: no cover
        _underthesea_word_tokenize = None


def normalize_text(text: str) -> str:
    """Chuẩn hóa NFC, lowercase, gọn khoảng trắng (mọi whitespace -> 1 space, strip)."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    return _WS_RE.sub(" ", text).strip()


def fold_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt: NFD + loại combining marks; "đ"->"d", "Đ"->"D".

    KHÔNG lowercase — hàm này chỉ bỏ dấu, caller tự normalize trước nếu cần.
    Lưu ý: "đ"/"Đ" không có canonical decomposition nên phải thay thủ công.
    """
    text = text.replace("đ", "d").replace("Đ", "D")  # đ -> d, Đ -> D
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def tokenize(text: str) -> list[str]:
    """Tách token: normalize_text rồi giữ chữ + số, bỏ dấu câu.

    Ưu tiên bộ tách từ ghép (pyvi/underthesea, nối "_" như "giá_vàng"); không có
    -> regex ``\\w+`` theo âm tiết. Trả list token lowercase, GIỮ NGUYÊN dấu.
    """
    norm = normalize_text(text)
    if not norm:
        return []
    if _pyvi_tokenize is not None:  # pragma: no cover - tùy môi trường
        try:
            # ViTokenizer nối từ ghép bằng "_" (vd "bất_động_sản")
            return _TOKEN_RE.findall(_pyvi_tokenize(norm))
        except Exception:
            pass  # lỗi runtime -> thử tiếp/fallback
    if _underthesea_word_tokenize is not None:  # pragma: no cover
        try:
            joined = _underthesea_word_tokenize(norm, format="text")
            return _TOKEN_RE.findall(joined)
        except Exception:
            pass
    return _TOKEN_RE.findall(norm)


def fold_tokens(tokens: list[str]) -> list[str]:
    """Bỏ dấu từng token (phục vụ matching không phân biệt dấu)."""
    return [fold_diacritics(tok) for tok in tokens]
