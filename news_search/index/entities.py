"""F-06 — Trích xuất thực thể (NER) + knowledge graph in-memory.

- ``extract_entities``: dùng ``underthesea.ner`` nếu import được (import guard);
  fallback heuristic thuần Python: cụm >=1 từ viết hoa liên tiếp, nhận diện chữ
  hoa CÓ DẤU tiếng Việt bằng ``str.istitle()/str.isupper()`` trên từng từ
  (KHÔNG dựa vào [A-Z] thuần ASCII), phân loại ORG/LOC/PER/MISC theo từ khóa.
- ``KnowledgeGraph``: đồ thị bài viết <-> thực thể trong bộ nhớ, match tên
  KHÔNG DẤU + không phân biệt hoa thường (tự fold bằng ``unicodedata`` nội bộ,
  không phụ thuộc module khác). Production thay bằng Neo4j.

Kết quả deterministic: không dùng hash() builtin, không phụ thuộc thời gian.
"""

from __future__ import annotations

import re
import unicodedata

from news_search.models import Entity

# --- Import guard cho underthesea (thư viện nặng, có thể không cài) ---
try:  # pragma: no cover - phụ thuộc môi trường
    from underthesea import ner as _underthesea_ner
except ImportError:  # pragma: no cover
    _underthesea_ner = None

# ---------------------------------------------------------------------------
# Fold nội bộ (bỏ dấu + lowercase) — KHÔNG import từ ingest.tokenizer
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _fold(text: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase + gọn khoảng trắng (chỉ dùng nội bộ)."""
    text = unicodedata.normalize("NFC", text).replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", stripped).strip().lower()


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?…;:\n\r]+")

# Cụm từ khóa (đã fold) chỉ tổ chức — xuất hiện trong chunk -> ORG
_ORG_PHRASES: tuple[tuple[str, ...], ...] = (
    ("cong", "ty"),
    ("tong", "cong", "ty"),
    ("tap", "doan"),
    ("ngan", "hang"),
    ("ubnd",),
    ("uy", "ban"),
    ("bo",),
    ("so",),
    ("truong",),
    ("dai", "hoc"),
    ("hoc", "vien"),
    ("vien",),
    ("benh", "vien"),
    ("hoi",),
    ("lien", "doan"),
    ("nha", "may"),
    ("cau", "lac", "bo"),
    ("cong", "an"),
    ("bao",),
)

# Cụm từ khóa (đã fold) chỉ địa danh — xuất hiện trong chunk -> LOC
_LOC_PHRASES: tuple[tuple[str, ...], ...] = (
    ("thanh", "pho"),
    ("tinh",),
    ("tp",),
    ("quan",),
    ("huyen",),
    ("phuong",),
    ("xa",),
    ("thi", "xa"),
    ("thi", "tran"),
    ("thu", "do"),
    ("song",),
    ("nui",),
    ("dao",),
    ("vinh",),
)

_ALL_PHRASES: tuple[tuple[str, ...], ...] = _ORG_PHRASES + _LOC_PHRASES
_ALL_PHRASES_SET = frozenset(_ALL_PHRASES)

# Token (fold) đứng NGAY TRƯỚC chunk gợi ý địa danh: "tại X", "ở X", "tỉnh X"...
_LOC_PRECEDING = frozenset(
    {"tai", "o", "tinh", "pho", "quan", "huyen", "phuong", "xa", "tp"}
)

# Stopword (fold) hay viết hoa đầu câu — bỏ khỏi đầu chunk nếu chunk mở đầu câu
_START_STOPWORDS = frozenset(
    {
        "ong", "ba", "anh", "chi", "em", "theo", "tai", "o", "trong", "ngoai",
        "tren", "duoi", "giua", "sau", "truoc", "khi", "ve", "voi", "tu", "den",
        "cho", "va", "neu", "vi", "de", "tuy", "nhung", "con", "la", "co",
        "khong", "chua", "da", "se", "vua", "moi", "cung", "nhu", "day", "cac",
        "mot", "hom", "ngay", "thang", "sang", "chieu", "toi", "dem", "thu",
        "chu", "pho", "tong", "giam", "viec", "nguoi", "chung", "ta", "hien",
        "luc", "gio", "dip", "nhan",
    }
)

# Địa danh quen thuộc (fold, so khớp TOÀN chunk) -> LOC
_LOC_GAZETTEER = frozenset(
    {
        "viet nam", "ha noi", "tp hcm", "tphcm", "sai gon", "da nang",
        "hai phong", "can tho", "hue", "thua thien hue", "da lat", "nha trang",
        "vung tau", "ba ria vung tau", "phu quoc", "dong thap", "long an",
        "an giang", "ben tre", "binh duong", "dong nai", "quang ninh",
        "nghe an", "thanh hoa", "lam dong", "khanh hoa", "kien giang",
        "ca mau", "bac ninh", "hai duong", "nam dinh", "thai binh",
        "quang nam", "quang ngai", "binh dinh", "phu yen", "gia lai",
        "dak lak", "son la", "lao cai", "yen bai", "cao bang", "lang son",
        "ha giang", "hoa binh", "vinh phuc", "hung yen", "ninh binh",
        "quang binh", "quang tri", "tay ninh", "my", "hoa ky", "trung quoc",
        "nhat ban", "han quoc", "thai lan", "lao", "campuchia", "singapore",
        "malaysia", "indonesia", "philippines", "nga", "phap", "duc", "anh",
        "y", "uc", "an do", "chau au", "chau a", "dong nam a",
    }
)


def _is_cap(word: str) -> bool:
    """Từ 'viết hoa' — dùng istitle/isupper (Unicode-aware, hỗ trợ Đ, Ạ, Ế...)."""
    return bool(word) and (word.istitle() or word.isupper() or word[0].isupper())


def _bridge_end(folded: list[str], chunk_start: int, j: int) -> int | None:
    """Tìm keyword phrase phủ vị trí từ thường ``j`` để 'bắc cầu' chunk.

    Ví dụ: "Tập đoàn Vingroup" — 'đoàn' (thường) nằm trong phrase ("tap","doan")
    bắt đầu tại từ 'Tập' đã thuộc chunk. Trả về index của token NGAY SAU phrase
    (token này phải viết hoa thì mới nối tiếp), hoặc None nếu không bắc cầu được.
    """
    n = len(folded)
    for phrase in _ALL_PHRASES:
        plen = len(phrase)
        for s in range(max(chunk_start, j - plen + 1), j + 1):
            if s + plen <= j or s + plen > n:
                continue  # phrase phải phủ vị trí j và không tràn câu
            if tuple(folded[s : s + plen]) == phrase:
                return s + plen
    return None


def _contains_phrase(
    f_words: tuple[str, ...], phrases: tuple[tuple[str, ...], ...]
) -> bool:
    """Chunk (đã fold) có chứa keyword phrase dạng dãy từ liên tiếp không."""
    for phrase in phrases:
        plen = len(phrase)
        for s in range(len(f_words) - plen + 1):
            if f_words[s : s + plen] == phrase:
                return True
    return False


def _classify(words: list[str], f_words: list[str], loc_hint: bool) -> str:
    """Phân loại chunk theo từ khóa mô tả trong CONTRACTS §13."""
    ft = tuple(f_words)
    if _contains_phrase(ft, _ORG_PHRASES):
        return "ORG"
    if loc_hint or _contains_phrase(ft, _LOC_PHRASES) or " ".join(f_words) in _LOC_GAZETTEER:
        return "LOC"
    n_cap = sum(1 for w in words if _is_cap(w))
    if 2 <= n_cap <= 4:
        return "PER"
    return "MISC"


def _extract_heuristic(text: str) -> list[Entity]:
    """Fallback NER: cụm từ viết hoa liên tiếp + bắc cầu qua keyword phrase."""
    entities: list[Entity] = []
    for sentence in _SENT_SPLIT_RE.split(text):
        if not sentence.strip():
            continue
        tokens = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(sentence)]
        folded = [_fold(w) for w, _, _ in tokens]
        n = len(tokens)

        def _adjacent(a: int, b: int) -> bool:
            # Giữa 2 token chỉ có khoảng trắng (dấu câu -> ngắt chunk)
            return sentence[tokens[a][2] : tokens[b][1]].strip() == ""

        i = 0
        while i < n:
            if not _is_cap(tokens[i][0]):
                i += 1
                continue
            # Mở rộng chunk từ i: từ viết hoa liền kề, hoặc bắc cầu keyword
            j = i + 1
            while j < n and _adjacent(j - 1, j):
                if _is_cap(tokens[j][0]):
                    j += 1
                    continue
                end = _bridge_end(folded, i, j)
                if (
                    end is not None
                    and end < n
                    and all(_adjacent(k - 1, k) for k in range(j + 1, end + 1))
                    and _is_cap(tokens[end][0])
                ):
                    j = end + 1  # nuốt phrase + từ viết hoa kế tiếp
                    continue
                break

            words = [tokens[k][0] for k in range(i, j)]
            f_words = list(folded[i:j])
            loc_hint = i > 0 and folded[i - 1] in _LOC_PRECEDING
            # Bỏ từ mở đầu câu nếu là stopword thường gặp (Ông, Bà, Theo, Tại...)
            if i == 0 and f_words and f_words[0] in _START_STOPWORDS:
                if f_words[0] in ("tai", "o"):
                    loc_hint = True
                words, f_words = words[1:], f_words[1:]
            i = j
            if not words:
                continue
            # Chunk chỉ gồm đúng một keyword ("Bộ", "Hội"...) -> không phải tên riêng
            if tuple(f_words) in _ALL_PHRASES_SET:
                continue
            name = " ".join(words)
            entities.append(Entity(name=name, type=_classify(words, f_words, loc_hint)))
    return entities


# ---------------------------------------------------------------------------
# Nhánh underthesea
# ---------------------------------------------------------------------------


def _map_label(label: str) -> str:
    """Map nhãn NER của underthesea về PER/ORG/LOC; nhãn khác -> MISC."""
    up = label.upper()
    for known in ("PER", "ORG", "LOC"):
        if up.startswith(known):
            return known
    return "MISC"


def _extract_with_underthesea(text: str) -> list[Entity]:  # pragma: no cover
    """Gom chuỗi nhãn BIO của underthesea.ner thành list Entity."""
    rows = _underthesea_ner(text)
    entities: list[Entity] = []
    cur_words: list[str] = []
    cur_type = ""

    def _flush() -> None:
        nonlocal cur_words, cur_type
        if cur_words:
            entities.append(Entity(name=" ".join(cur_words), type=cur_type))
        cur_words, cur_type = [], ""

    for row in rows:
        word, tag = str(row[0]), str(row[-1])
        if tag.startswith("B-"):
            _flush()
            cur_type = _map_label(tag[2:])
            cur_words = [word]
        elif tag.startswith("I-") and cur_words:
            cur_words.append(word)
        else:
            _flush()
    _flush()
    return entities


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[Entity]:
    """Trích xuất thực thể từ văn bản (F-06).

    Dùng underthesea nếu có; nếu không (hoặc lỗi runtime) dùng heuristic
    thuần Python. Dedup theo (name, type), giữ thứ tự xuất hiện đầu tiên.
    """
    if not text or not text.strip():
        return []
    text = unicodedata.normalize("NFC", text)
    raw: list[Entity]
    if _underthesea_ner is not None:
        try:  # pragma: no cover - chỉ chạy khi có underthesea
            raw = _extract_with_underthesea(text)
        except Exception:
            raw = _extract_heuristic(text)
    else:
        raw = _extract_heuristic(text)

    seen: set[tuple[str, str]] = set()
    out: list[Entity] = []
    for ent in raw:
        if not _keep_entity(ent):
            continue
        key = (ent.name, ent.type)
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
    return out


def _keep_entity(ent: Entity) -> bool:
    """Lọc nhiễu: bỏ MISC một-từ dạng title-case (thường là từ đầu câu bị nhận
    nhầm: "Phân", "Kết", "Giới"...). Vẫn giữ acronym viết hoa toàn bộ (UNESCO,
    WHO, UBND) vì đó là thực thể thật."""
    if ent.type == "MISC" and len(ent.name.split()) == 1 and not ent.name.isupper():
        return False
    return True


class KnowledgeGraph:
    """Đồ thị tri thức in-memory: bài viết <-> thực thể (production: Neo4j).

    Thực thể được khóa theo (tên đã fold, type) nên "Hà Nội" / "ha noi" /
    "HÀ NỘI" cùng trỏ về một node; bản hiển thị là Entity xuất hiện đầu tiên.
    """

    def __init__(self) -> None:
        # article_id -> list Entity (đã dedup theo khóa fold, giữ thứ tự)
        self._article_entities: dict[str, list[Entity]] = {}
        # (folded_name, type) -> set article_id
        self._entity_articles: dict[tuple[str, str], set[str]] = {}
        # (folded_name, type) -> Entity đại diện (dạng gốc thấy đầu tiên)
        self._display: dict[tuple[str, str], Entity] = {}
        # folded_name -> set khóa (folded_name, type) — tra cứu theo tên
        self._name_keys: dict[str, set[tuple[str, str]]] = {}

    @staticmethod
    def _key(entity: Entity) -> tuple[str, str]:
        return (_fold(entity.name), entity.type)

    def add_article(self, article_id: str, entities: list[Entity]) -> None:
        """Gắn thực thể cho bài viết; add lại cùng id = replace toàn bộ."""
        self.remove_article(article_id)
        deduped: list[Entity] = []
        seen: set[tuple[str, str]] = set()
        for ent in entities:
            key = self._key(ent)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ent)
        self._article_entities[article_id] = deduped
        for ent in deduped:
            key = self._key(ent)
            self._entity_articles.setdefault(key, set()).add(article_id)
            self._display.setdefault(key, ent)
            self._name_keys.setdefault(key[0], set()).add(key)

    def remove_article(self, article_id: str) -> None:
        """Gỡ bài khỏi đồ thị; id không tồn tại -> no-op."""
        ents = self._article_entities.pop(article_id, None)
        if not ents:
            return
        for ent in ents:
            key = self._key(ent)
            ids = self._entity_articles.get(key)
            if ids is None:
                continue
            ids.discard(article_id)
            if not ids:  # node thực thể mồ côi -> dọn sạch
                del self._entity_articles[key]
                self._display.pop(key, None)
                name_keys = self._name_keys.get(key[0])
                if name_keys is not None:
                    name_keys.discard(key)
                    if not name_keys:
                        del self._name_keys[key[0]]

    def entities_for_article(self, article_id: str) -> list[Entity]:
        """List thực thể của bài; bài không có -> []."""
        return list(self._article_entities.get(article_id, []))

    def articles_for_entity(self, name: str) -> set[str]:
        """Các bài chứa thực thể tên ``name`` — match không dấu, không hoa/thường."""
        folded = _fold(name)
        result: set[str] = set()
        for key in self._name_keys.get(folded, ()):  # gộp mọi type trùng tên
            result |= self._entity_articles.get(key, set())
        return result

    def top_entities(self, n: int = 10) -> list[tuple[Entity, int]]:
        """Top n thực thể theo SỐ BÀI chứa nó, giảm dần; tie-break deterministic."""
        ranked = sorted(
            self._entity_articles.items(),
            key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]),
        )
        return [(self._display[key], len(ids)) for key, ids in ranked[:n]]
