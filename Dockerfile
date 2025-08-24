# ---- base: python 3.7 ----
FROM python:3.7-slim-bullseye

# If Debian mirrors 404, uncomment below:
# RUN sed -i 's|deb.debian.org|archive.debian.org|g; s|security.debian.org|archive.debian.org|g' /etc/apt/sources.list \
#  && sed -i 's|^deb.*-security.*||g' /etc/apt/sources.list

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git tini curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

# non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# optional (for local docs only; Render ignores EXPOSE)
EXPOSE 8501

# Healthcheck must use $PORT (Render sets it). Use shell form so env expands.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD bash -lc 'curl -fsS http://127.0.0.1:${PORT:-8501}/_stcore/health || exit 1'

# Run with tini, bind to $PORT on Render; fallback 8501 for local
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
