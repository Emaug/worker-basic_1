import sys
import torch
import runpod
import time
from transformers import pipeline

def handler(event):
    print("Worker Start")
    
    input = event['input']
    prompt = input.get('prompt')
    seconds = input.get('seconds', 0)

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    time.sleep(seconds)

    print("Hello from your custom Runpod template!")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Initialize model
    print("\nLoading sentiment analysis model...")
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

    print("Model loaded successfully!")

    result = classifier(prompt)

    print("Inference done!")

    return result


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
