import sys
import torch
import runpod
import time
from transformers import pipeline
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import login
import os
import base64
from io import BytesIO

def handler(event):
    key = os.getenv("MY_KEY")
    print("ENV DEBUG:", key)
    login(token=key)

    # modell
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # ladda pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    print(f"Worker Start")
    input = event['input']

    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")

    time.sleep(seconds)  

    # ====== ENDA TILLAGDA DELEN ======
    print("Generating image...")
    image = pipe(prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": image_base64
    }


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
