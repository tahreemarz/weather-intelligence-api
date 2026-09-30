# Weather Intelligence API - production image for Render (or any Docker host)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELS_DIR=/app/models \
    PORT=8000

WORKDIR /app

# xgboost needs libgomp at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached layer): public packages from PyPI, then the custom package from TestPyPI.
# --no-deps keeps pip from resolving numpy/pandas/... against TestPyPI's incomplete index.
ARG PACKAGE_SPEC="amla-package-26016360>=0.2.0"
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip install --no-deps --index-url https://test.pypi.org/simple/ "${PACKAGE_SPEC}" \
 && python -c "import amla_package_26016360 as a, amla_package_26016360.weather; print('amla-package-26016360', a.__version__)"

# Application code and model artefacts
COPY app ./app
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s CMD python -c "import urllib.request,sys,os; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

# Render injects $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
