"""News Search Engine — hệ thống tìm kiếm cho trang báo chí số.

Kiến trúc phễu nhiều tầng (xem BaoCao-ThietKe.md):
    ingest -> lexical BM25 + semantic vector -> RRF fusion
           -> time-decay -> rerank -> dedup/MMR -> JSON 5 trường
"""

__version__ = "0.1.0"
