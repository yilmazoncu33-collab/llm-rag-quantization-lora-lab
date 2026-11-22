FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine.py app.py test_app.py ./

# Konteyner imajı inşa edilirken otomatik birim testlerini çalıştır
RUN pytest -v

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
