# News Search Engine — image demo NHẸ (chỉ core deps, KHÔNG torch/model nặng).
# Chạy offline tức thì với backend local/hash + tự nạp bài mẫu. Nâng cấp
# (vi/bge/milvus/opensearch/redis) cài thêm qua extras — xem README.
FROM python:3.11-slim

# Không ghi .pyc, log không buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Chỉ cài phụ thuộc LÕI (có sẵn wheel, không cần compiler): fastapi, uvicorn,
# pydantic, numpy, python-dotenv. KHÔNG cài requirements.txt (kéo theo torch).
RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn>=0.30" "pydantic>=2.7" "numpy>=1.26" "python-dotenv>=1.0"

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
    && mkdir -p /app/data/feedback && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=4s --start-period=8s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "news_search.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
