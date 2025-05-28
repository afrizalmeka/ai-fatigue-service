FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir websockets==11.0

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
