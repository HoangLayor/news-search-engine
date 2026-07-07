# News Search Engine — image demo NHẸ (chỉ core deps, KHÔNG torch/model nặng).
# Chạy offline tức thì với backend local/hash + tự nạp bài mẫu. Nâng cấp
# (vi/bge/milvus/opensearch/redis) cài thêm qua extras — xem README.
FROM python:3.11-slim

# Không ghi .pyc, log không buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Phụ thuộc LÕI (có sẵn wheel, không cần compiler) + pymilvus/redis cho các
# profile Milvus/Redis (đều lazy-import: KHÔNG kéo theo torch, không ảnh hưởng
# demo local mặc định). KHÔNG cài requirements.txt (nó kéo theo torch).
# Cài riêng PyTorch bản CPU siêu nhẹ
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài lõi + sentence-transformers + phụ thuộc BẮT BUỘC của tokenizer model VN.
# sentencepiece + protobuf: cần cho tokenizer AITeamVN (nền XLM-RoBERTa); thiếu ->
# transformers rơi về TikToken và CRASH khi nạp EMBEDDER/RERANKER=vi.
# pyvi: cho TOKENIZER=auto (tách từ ghép tiếng Việt).
RUN pip install --no-cache-dir \
    "fastapi>=0.115" "uvicorn>=0.30" "pydantic>=2.7" "numpy>=1.26" "python-dotenv>=1.0" \
    "pymilvus>=2.4" "redis>=5.0" "sentence-transformers" "datasketch" \
    "sentencepiece" "protobuf" "pyvi"

# Copy mã nguồn (bao gồm data/sample_articles.json, news_search/api/ui.html)
COPY news_search/ ./news_search/
COPY data/ ./data/
COPY pyproject.toml README.md ./

# Cấu hình mặc định DEMO: offline, local/hash, tự nạp mẫu, bật feedback/cache/UI.
# Ghi đè bất kỳ biến nào qua `docker run -e` hoặc docker-compose.
ENV LEXICAL_BACKEND=local \
    VECTOR_BACKEND=local \
    EMBEDDER=hash \
    RERANKER=none \
    TOKENIZER=regex \
    DEDUP_BACKEND=local \
    SOURCE=sample \
    AUTOLOAD_SAMPLE=true \
    FEEDBACK_ENABLED=true \
    CACHE_ENABLED=true \
    METRICS_ENABLED=true \
    UI_ENABLED=true

# Chạy bằng user không phải root (bảo mật)
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/feedback /app/.venv/huggingface_cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=4s --start-period=8s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "news_search.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
