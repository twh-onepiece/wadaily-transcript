FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    websockets \
    scipy

# モデル事前ダウンロード
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='kotoba-tech/kotoba-whisper-v2.0')"

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

