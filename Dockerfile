FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 scanner \
    && chown -R scanner:scanner /app

USER scanner

EXPOSE 10000

CMD ["sh", "-c", "python run_with_dashboard.py --dry-run --port ${PORT:-10000}"]
