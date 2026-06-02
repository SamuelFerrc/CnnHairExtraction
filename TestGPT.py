import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm.auto import tqdm

# =========================
# CONFIG
# =========================

input_path = "outputs/colored_lines.png"   # ou "colored_lines.png"
output_path = "graph.png"

# modelo mais realista que SD 1.5 puro
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# =========================
# LOAD PIPELINE
# =========================

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False
)

pipe = pipe.to(device)

if device == "cuda":
    pipe.enable_attention_slicing()

# =========================
# INPUT IMAGE
# =========================

image = Image.open(input_path).convert("RGB")
image = image.resize((768, 512))

# =========================
# PROMPTS
# =========================

prompt = """
photorealistic long brown human hair,
only hair,
dense natural hair strands,
realistic hair fibers,
smooth flowing hairstyle,
dark brown glossy hair,
soft specular highlights,
realistic layered hair,
individual fine strands,
black background,
studio lighting,
high detail,
sharp focus
"""

negative_prompt = """
drawing,
sketch,
lineart,
skeleton,
wireframe,
white lines,
colored lines,
cartoon,
anime,
painting,
fake hair,
plastic hair,
broken strands,
disconnected strands,
messy artifacts,
face,
head,
skin,
eyes,
mouth,
neck,
body,
blurry,
low quality,
deformed,
distorted
"""

# =========================
# PROGRESS BAR
# =========================

steps = 50
pbar = tqdm(total=steps)

def callback(pipe, step_index, timestep, callback_kwargs):
    pbar.update(1)
    return callback_kwargs

# =========================
# GENERATE
# =========================

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,

    # Aumente para transformar mais em cabelo real
    strength=0.88,

    # Força do prompt
    guidance_scale=9.0,

    num_inference_steps=steps,
    callback_on_step_end=callback
).images[0]

pbar.close()

# =========================
# SAVE
# =========================

result.save(output_path)
print("Salvo em:", output_path)
