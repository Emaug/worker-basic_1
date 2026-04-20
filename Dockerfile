FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN python -c "from diffusers import StableDiffusionXLPipeline; StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')"
COPY rp_handler.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]
