FROM runpod/pytorch:3.10-2.1.2-cuda12.1

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
