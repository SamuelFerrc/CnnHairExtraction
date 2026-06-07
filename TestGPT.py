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

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Device:", device)
print("Torch dtype:", dtype)

lineart = Image.open(input_path).convert("RGB")
lineart = lineart.resize((1024, 1024))

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=dtype
)

vae = AutoencoderKL.from_pretrained(
    vae_model,
    torch_dtype=dtype
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=dtype
)

pipe = pipe.to(device)

if device == "cuda":
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

prompt = """
only realistic hair strands, isolated hair, no face, no eyes, no skin,
no body, no head, no person, white background, detailed hair fibers,
natural hair flow, clean hair shape
"""

negative_prompt = """
face, head, person, body, eyes, mouth, nose, skin, portrait, anime,
cartoon, blurry, low quality, extra objects, background, clothes
"""

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=lineart,
    num_inference_steps=25,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.9,
    width=1024,
    height=1024
).images[0]

result.save(output_path)

print("Imagem salva em:", output_path)
