"""Chuẩn hóa & token hóa văn bản tiếng Việt (F-03).

Ưu tiên underthesea (word_tokenize) nếu cài được; fallback thuần stdlib
tách token bằng regex ``\\w+`` (unicode). Mọi hàm deterministic, không I/O.
"""

from __future__ import annotations

import re
import unicodedata

# --- Import guard: underthesea là thư viện nặng, không bắt buộc ---
try:  # pragma: no cover - phụ thuộc môi trường
    from underthesea import word_tokenize as _underthesea_word_tokenize
except ImportError:  # pragma: no cover
    _underthesea_word_tokenize = None

# Gom mọi whitespace (space, tab, newline, nbsp...) về 1 space
_WS_RE = re.compile(r"\s+")
# Token = chuỗi chữ/số/underscore (unicode); dấu câu bị loại
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


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

    Có underthesea: dùng word_tokenize (từ ghép nối bằng "_") trên văn bản
    đã normalize; không có: tách theo regex ``\\w+`` (unicode).
    Trả về list token lowercase, GIỮ NGUYÊN dấu tiếng Việt.
    """
    norm = normalize_text(text)
    if not norm:
        return []
    if _underthesea_word_tokenize is not None:  # pragma: no cover - tùy môi trường
        try:
            # format="text": từ ghép được nối bằng "_" (vd "giá_vàng")
            joined = _underthesea_word_tokenize(norm, format="text")
            # \w+ giữ nguyên "_" trong từ ghép, loại dấu câu đứng riêng
            return _TOKEN_RE.findall(joined)
        except Exception:
            pass  # underthesea lỗi runtime -> rơi về fallback
    return _TOKEN_RE.findall(norm)


def fold_tokens(tokens: list[str]) -> list[str]:
    """Bỏ dấu từng token (phục vụ matching không phân biệt dấu)."""
    return [fold_diacritics(tok) for tok in tokens]
