# Đặt tại gốc dự án để pytest thêm thư mục này vào sys.path,
# cho phép tests import trực tiếp package `news_search`.
#
# Ép backend về local/hash TRƯỚC khi import config để test luôn:
#   - hermetic (không cần Milvus/Postgres/mạng),
#   - nhanh & deterministic (không nạp model BGE ~2GB).
# config.py dùng load_dotenv(override=False) nên các giá trị đặt ở đây thắng .env.
import os

os.environ["VECTOR_BACKEND"] = "local"
os.environ["LEXICAL_BACKEND"] = "local"
os.environ["EMBEDDER"] = "hash"
os.environ["RERANKER"] = "none"
os.environ["SOURCE"] = "sample"
os.environ["TOKENIZER"] = "regex"          # tách từ deterministic (không phụ thuộc pyvi)
os.environ["DEDUP_BACKEND"] = "local"      # dedup thuần Python (deterministic)
