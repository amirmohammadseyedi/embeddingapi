FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface U2NET_HOME=/cache/rembg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip check
COPY app ./app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
