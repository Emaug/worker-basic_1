import os, base64
from io import BytesIO

import torch
import runpod
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import login

HF_TOKEN = os.getenv("MY_KEY")
if HF_TOKEN:
    login(token=HF_TOKEN)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Choose ONE:
pipe = pipe.to("cuda")  # if you have VRAM
# OR: pipe.enable_model_cpu_offload()

def handler(event):
    inp = event.get("input", {})
    prompt = inp.get("prompt", "a futuristic city at sunset")

    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    return {"png_base64": base64.b64encode(buf.getvalue()).decode()}

runpod.serverless.start({"handler": handler})
