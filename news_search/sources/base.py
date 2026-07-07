"""Hợp đồng chung cho mọi nguồn dữ liệu bài viết.

``RawArticle`` là dict thô có các khóa tương thích
:func:`news_search.ingest.cleaner.normalize_article`:

    article_id, title, body, url, published_at, author, category, source, status, tags

Nguồn chỉ chịu trách nhiệm **lấy & ánh xạ** dữ liệu về dạng dict trên; việc làm
sạch HTML / chuẩn hóa datetime do tầng ingest lo. Tách bạch này giúp thay nguồn
mà không đụng tới index/search.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional, Protocol, runtime_checkable

# dict thô; dùng alias cho ý nghĩa rõ ràng ở chữ ký hàm
RawArticle = dict


@runtime_checkable
class ArticleSource(Protocol):
    """Giao diện nguồn dữ liệu (streaming để chịu được kho lớn).

    Mọi phương thức fetch trả về **iterator** dict thô — nguồn nên stream theo lô
    thay vì nạp toàn bộ vào RAM.
    """

    def count(self) -> int:
        """Tổng số bài đủ điều kiện index (đã xuất bản, chưa xóa)."""
        ...

    def fetch_all(self, batch_size: int = 500) -> Iterator[RawArticle]:
        """Duyệt toàn bộ bài viết đủ điều kiện."""
        ...

    def fetch_since(self, since: datetime, batch_size: int = 500) -> Iterator[RawArticle]:
        """Duyệt các bài cập nhật/xuất bản từ mốc ``since`` (phục vụ index tăng dần)."""
        ...

    def fetch_by_ids(self, ids: list[str]) -> Iterator[RawArticle]:
        """Lấy các bài theo danh sách id (phục vụ re-index có chọn lọc)."""
        ...

    def close(self) -> None:
        """Giải phóng tài nguyên (đóng kết nối...). An toàn khi gọi nhiều lần."""
        ...


def batched(iterable, size: int) -> Iterator[list]:
    """Gom một iterable thành các lô ``size`` phần tử (tiện ích cho nguồn)."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
