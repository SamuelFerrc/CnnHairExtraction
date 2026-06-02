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
Generate ONLY the hair structure represented by the provided drawing.

Use the input drawing strictly as the sole geometric and topological reference.

The generated result must follow:
- the exact flow of the strands,
- the exact orientation,
- the exact silhouette,
- the exact topology,
- the exact curvature,
- the exact strand direction
present in the drawing.

Do NOT invent:
- faces,
- heads,
- people,
- bodies,
- skin,
- backgrounds,
- extra hairstyles,
- accessories,
- additional geometry.

The output must contain ONLY isolated realistic hair generated directly from the input sketch.

Interpret the drawing as a sparse hair topology map and reconstruct a dense realistic hair structure from it.

Preserve:
- strand continuity,
- strand alignment,
- hair flow,
- local topology,
- strand grouping.

Black background.
Only hair.
Photorealistic hair fibers.
Dense coherent hair strands.
"""

negative_prompt = """
person,
human,
woman,
man,
face,
eyes,
nose,
mouth,
skin,
head,
portrait,
body,
shoulders,
background,
clothes,
accessories,
hat,
ears,
neck,
anime,
cartoon,
painting,
illustration,
drawing,
sketch,
lineart,
wireframe,
extra hair,
new hairstyle,
topology changes,
different orientation,
messy strands,
broken strands,
random hair,
floating hair,
deformed hair,
artifact,
blurry
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
