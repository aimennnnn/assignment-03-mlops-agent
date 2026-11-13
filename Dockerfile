# Image of the prediction API (api/main.py) with the exported champion model inside.
#   python -m training.export_model        (creates api/model/ first)
#   docker build -t cancer-api:1.0.0 .
FROM python:3.13-slim

WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/main.py .
COPY api/model/ ./model/

# Do not run as root inside the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
