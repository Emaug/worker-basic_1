import torch
import runpod
import os
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import login

# 🔐 login med env variabel
HF_TOKEN = os.getenv("MY_KEY")
login(token=HF_TOKEN)

# 🔥 ladda modell EN gång (globalt = snabbare)
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.enable_model_cpu_offload()

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)


def handler(event):
    input = event["input"]
    prompt = input.get("prompt", "a futuristic city at sunset")

    print(f"Generating: {prompt}")

    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # spara temporärt
    output_path = "/tmp/output.png"
    image.save(output_path)

    return {
        "message": "done",
        "image_path": output_path
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
