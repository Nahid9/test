FROM python:3.9.20-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip==20.3.4 && \
    pip install tensorflow==2.10.0 -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
