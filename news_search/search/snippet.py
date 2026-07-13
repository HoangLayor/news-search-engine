"""Trích đoạn nổi bật — snippet có bôi đậm từ khớp (F-15).

Chọn cửa sổ ~max_len ký tự chứa nhiều token khớp nhất; match không dấu,
không phân biệt hoa thường, so trên từ nguyên vẹn; bôi đậm bằng <b>...</b>
nhưng GIỮ NGUYÊN chữ gốc có dấu của body. Chỉ dùng stdlib.
"""

from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ELLIPSIS = "..."


def _fold(text: str) -> str:
    """Bỏ dấu tiếng Việt: NFD + bỏ combining marks; đ/Đ -> d (không lowercase)."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("đ", "d").replace("Đ", "d")


def _query_token_set(query_tokens: list[str]) -> set[str]:
    """Chuẩn hóa token truy vấn thành tập TỪ ĐƠN (fold + lowercase).

    Body được match theo từng từ nên tập truy vấn phải là từ đơn. Tách theo cả
    khoảng trắng lẫn '_' để bền vững khi caller truyền cụm nhiều từ ("lam phat")
    hoặc từ ghép do underthesea nối bằng '_' ("lam_phat").
    """
    result: set[str] = set()
    for token in query_tokens:
        folded = _fold(token).lower()
        for part in re.split(r"[\s_]+", folded):
            if part:
                result.add(part)
    return result


def make_snippet(body: str, query_tokens: list[str], max_len: int = 200) -> str:
    """Sinh snippet ~max_len ký tự, bôi đậm từ khớp truy vấn bằng <b>...</b>.

    - Cửa sổ được chọn là đoạn chứa NHIỀU từ khớp nhất (match không dấu,
      không phân biệt hoa thường, trên từ nguyên vẹn).
    - Thêm "..." đầu/cuối nếu cắt giữa văn bản.
    - Không có match -> đầu body cắt max_len. body rỗng -> "".
    """
    if not body:
        return ""
    # NFC để offset ổn định và \w khớp đúng chữ có dấu
    body = unicodedata.normalize("NFC", body)
    qset = _query_token_set(query_tokens)

    # Tìm mọi từ trong body khớp token truy vấn (so trên bản folded+lower)
    matches: list[tuple[int, int]] = []  # (start, end) theo ký tự
    if qset:
        for m in _WORD_RE.finditer(body):
            if _fold(m.group()).lower() in qset:
                matches.append((m.start(), m.end()))

    if not matches:
        snippet = body[:max_len]
        return snippet + (_ELLIPSIS if len(body) > max_len else "")

    # Chọn cửa sổ max_len chứa nhiều match nhất (quét mỗi match làm mốc đầu)
    best_i, best_j, best_count = 0, 0, 0
    for i, (start_i, _) in enumerate(matches):
        j = i
        count = 0
        for k in range(i, len(matches)):
            if matches[k][1] <= start_i + max_len:
                j = k
                count += 1
            else:
                break
        if count > best_count:  # tie -> giữ cửa sổ sớm nhất (deterministic)
            best_i, best_j, best_count = i, j, count

    first_start = matches[best_i][0]
    last_end = matches[best_j][1]
    # Căn giữa cụm match trong cửa sổ để có ngữ cảnh hai phía
    slack = max(0, max_len - (last_end - first_start))
    win_start = max(0, first_start - slack // 2)
    win_end = min(len(body), win_start + max_len)
    win_start = max(0, win_end - max_len)  # tận dụng hết max_len nếu chạm cuối

    # Ghép snippet, chèn <b>...</b> quanh match nằm TRỌN trong cửa sổ
    parts: list[str] = []
    cursor = win_start        
    for start, end in matches:
        if start < win_start or end > win_end:
            continue
        parts.append(body[cursor:start])
        parts.append(f"<b>{body[start:end]}</b>")
        cursor = end
    parts.append(body[cursor:win_end])
    snippet = "".join(parts)

    prefix = _ELLIPSIS if win_start > 0 else ""
    suffix = _ELLIPSIS if win_end < len(body) else ""
    return f"{prefix}{snippet}{suffix}"
