import sys
import torch
import runpod
import time
from transformers import pipeline

def handler(event):
    try:
        print("Worker Start")
        
        input = event['input']
        prompt = input.get('prompt')
        seconds = input.get('seconds', 0)

        print(f"Received prompt: {prompt}")
        print(f"Sleeping for {seconds} seconds...")
        time.sleep(seconds)

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

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
