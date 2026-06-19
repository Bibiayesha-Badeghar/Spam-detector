# ── Base image ──────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Dependencies ────────────────────────────────────────────
WORKDIR /app

# Install dependencies first (layer caching: only re-runs when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────
COPY . .

# Train the model so it's baked into the image and ready on startup
RUN python train_model.py

# ── Runtime ─────────────────────────────────────────────────
# Run as non-root for security
RUN adduser --disabled-password --no-create-home appuser
USER appuser

EXPOSE 5000

CMD ["python", "app.py"]
