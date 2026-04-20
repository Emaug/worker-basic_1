import runpod
import torch
from diffusers import StableDiffusionXLPipeline

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("Loading SDXL model...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
    return pipe

def handler(event):
    input_data = event["input"]
    prompt = input_data.get("prompt", "a photo of a cat")
    steps = input_data.get("steps", 30)
    guidance = input_data.get("guidance", 7.5)
    seed = input_data.get("seed", None)

    pipe = load_model()

    generator = torch.Generator("cuda")
    if seed is not None:
        generator.manual_seed(seed)

    print(f"Generating image for prompt: {prompt}")

    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator
    ).images[0]

    output_path = "/tmp/output.png"
    image.save(output_path)

    return {
        "prompt": prompt,
        "image_path": output_path
    }

runpod.serverless.start({"handler": handler})
