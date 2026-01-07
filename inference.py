from diffusers import StableDiffusionPipeline
import torch
 
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
lora_weights = "./dreambooth_diandian"
 
pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float16)
pipe.unet.load_attn_procs(lora_weights)  # 加载LoRA权重
pipe.to("cuda")
 
prompt = input("input your prompt: ")
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("generated_image.png")