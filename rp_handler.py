import sys
import torch
import runpod
import time  
import time
import signal
from transformers import pipeline

def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
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
    
    # MODEL LOADING OPTIONS:
    
    # OPTION 1: From Hugging Face Hub cache (default)
    # Bakes the model into the container image using transformers pipeline
    # Behavior: Loads model from the cache, requires local_files_only=True
    classifier = pipeline(
        "sentiment-analysis",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        device=device,
        model_kwargs={"local_files_only": True},
    )

    print("Model loaded successfully!")
    
    # Example inference
   return prompt


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
