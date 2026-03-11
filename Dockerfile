FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the reranker model so it's baked into the image
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
