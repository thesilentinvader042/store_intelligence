FROM python:3.11-slim

WORKDIR /project

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/project/app:/project/pipeline"

# Railway injects $PORT. Locally it defaults to 8000.
# railway.toml overrides CMD with the $PORT-aware version for Railway.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
