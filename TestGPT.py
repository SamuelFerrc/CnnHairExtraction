import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

from PIL import Image
from tqdm.auto import tqdm

# =========================================================
# CONFIG
# =========================================================

input_path = "outputs/colored_lines.png"
output_path = "graph.png"

# MODELO REALISTA
base_model = "SG161222/Realistic_Vision_V5.1_noVAE"

# CONTROLNET PARA LINEART / STRUCTURE
controlnet_model = "lllyasviel/control_v11p_sd15_lineart"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# =========================================================
# LOAD CONTROLNET
# =========================================================

controlnet = ControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=dtype
)

# =========================================================
# LOAD PIPELINE
# =========================================================

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe = pipe.to(device)

if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_attention_slicing()

# =========================================================
# INPUT IMAGE
# =========================================================

image = Image.open(input_path).convert("RGB")

# resolução maior preserva melhor a topologia
image = image.resize((1024, 1024))

# =========================================================
# PROMPT
# =========================================================

prompt = """
Generate realistic hair STRICTLY from the provided drawing.

Use ONLY the drawing as structural reference.

Preserve EXACTLY:
- strand flow
- strand direction
- topology
- silhouette
- curvature
- strand grouping
- hair distribution

Do not modify the geometry.

Do not create new hairstyle shapes.

Transform the white lines into realistic hair fibers while keeping the same exact structure.

Only isolated hair.

No human.

No face.

No body.

No scalp.

No portrait.

Black background.

Photorealistic hair strands.

Dense coherent hair fibers.

Ultra detailed realistic hair texture.
"""

negative_prompt = """
person,
human,
woman,
man,
face,
portrait,
head,
skin,
eyes,
nose,
mouth,
ears,
neck,
body,
shoulders,
clothes,
background,
environment,
anime,
cartoon,
painting,
drawing,
sketch,
illustration,
wireframe,
lineart,
extra strands,
new hairstyle,
topology change,
different shape,
different direction,
different curvature,
messy hair,
floating strands,
artifact,
blurry,
deformed,
fake hair
"""

# =========================================================
# PROGRESS BAR
# =========================================================

steps = 50

pbar = tqdm(total=steps)

def callback(pipe, step_index, timestep, callback_kwargs):
    pbar.update(1)
    return callback_kwargs

# =========================================================
# GENERATE
# =========================================================

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,

    # IMAGEM ESTRUTURAL
    image=image,

    # força do ControlNet
    controlnet_conditioning_scale=1.7,

    # força do prompt
    guidance_scale=8.5,

    # passos diffusion
    num_inference_steps=steps,

    # preserva mais a estrutura
    strength=0.55,

    callback_on_step_end=callback
).images[0]

pbar.close()

# =========================================================
# SAVE
# =========================================================

result.save(output_path)

print("Salvo em:", output_path)
