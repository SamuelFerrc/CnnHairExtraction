import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL
)

input_path = "outputs/colored_lines.png"
output_path = "outputs/hair_result.png"

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_model = "ShermanG/ControlNet-Standard-Lineart-for-SDXL"
vae_model = "madebyollin/sdxl-vae-fp16-fix"

num_steps = 30

lineart = Image.open(input_path).convert("RGB")
lineart = lineart.resize((1024, 1024))

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=torch.float16
)

vae = AutoencoderKL.from_pretrained(
    vae_model,
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16"
)

pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

def progress_callback(pipe, step_index, timestep, callback_kwargs):
    percent = (step_index + 1) / num_steps * 100
    print(f"\rProgresso: {percent:.1f}% ({step_index + 1}/{num_steps})", end="")
    return callback_kwargs

prompt = """
realistic human hair,
detailed individual hair strands,
natural hair flow,
hair only,
isolated hair object,
white background,
high detail,
soft lighting
"""

negative_prompt = """
face,
head,
body,
person,
skin,
eyes,
mouth,
nose,
portrait,
cartoon,
anime,
low quality,
blurry,
deformed,
extra objects,
background objects
"""

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=lineart,
    controlnet_conditioning_scale=0.85,
    guidance_scale=7.5,
    num_inference_steps=num_steps,
    width=1024,
    height=1024,
    callback_on_step_end=progress_callback
)

print("\nConcluído!")

image = result.images[0]
image.save(output_path)

print(f"Imagem salva em: {output_path}")