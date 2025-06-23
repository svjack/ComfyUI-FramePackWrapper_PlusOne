#### model: https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_T2V_14B_FusionX_LoRA.safetensors

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Wan2_1-T2V-14B_fp8_e4m3fn.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'ani_bright_landscape_w14_lora-step00005500.safetensors', 1)
    model = LoraLoaderModelOnly(model, 'Wan2.1_T2V_14B_FusionX_LoRA.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('anime style, a digital illustration video about ,Sunlight filters through clouds onto a cherry blossom-filled meadow, where a blue-haired youth in an indigo coat stands by a vermilion bridge, holding a book as petals drift past his smile. The scene blends dynamic nature with human stillness, using pink, red and blue hues to create harmony, while symbolic elements like blossoms and books add depth.', clip)
    conditioning2 = CLIPTextEncode('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', clip)
    latent = EmptyHunyuanLatentVideo(1024, 576, 81, 1)
    latent = KSampler(model, 1002959892432282, 10, 1, 'euler', 'beta', conditioning, conditioning2, latent, 1)
    vae = VAELoader('wan_2.1_vae.safetensors')
    image = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image, 16, 0, 'Fusionxt2v/FusionX', 'video/h264-mp4', False, True, None, None, None)

vim run_wan_fusionx_batch.py

import os
import time
import subprocess
from pathlib import Path
from datasets import load_dataset

# Configuration
SEED = 1002959892432282
OUTPUT_DIR = 'ComfyUI/output/Fusionxt2v'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def get_latest_mp4_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_mp4(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
    timeout = 6000  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_count = get_latest_mp4_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def load_prompts():
    """Load prompts from Hugging Face dataset"""
    dataset = load_dataset("svjack/Anime_Landscape_Images_Captioned")
    return [item['prompt'] for item in dataset['train']]

def generate_script(prompt, seed):
    """Generate the ComfyUI script with the given prompt and seed"""
    negative_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Wan2_1-T2V-14B_fp8_e4m3fn.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'ani_bright_landscape_w14_lora-step00005500.safetensors', 1)
    model = LoraLoaderModelOnly(model, 'Wan2.1_T2V_14B_FusionX_LoRA.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    conditioning2 = CLIPTextEncode('{negative_prompt}', clip)
    latent = EmptyHunyuanLatentVideo(1024, 576, 81, 1)
    latent = KSampler(model, {seed}, 10, 1, 'euler', 'beta', conditioning, conditioning2, latent, 1)
    vae = VAELoader('wan_2.1_vae.safetensors')
    image = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image, 16, 0, 'Fusionxt2v/FusionX', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def main():
    SEED = 1002959892432282
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load prompts from dataset
    print("Loading prompts from Hugging Face dataset...")
    prompts = load_prompts()
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts.")
    
    # Process each prompt
    for idx, prompt in enumerate(prompts):
        print(f"Processing prompt {idx + 1}/{total_prompts}")
        print(f"Prompt: {prompt[:100]}...")  # Show first 100 chars
        
        # Get current MP4 count
        initial_count = get_latest_mp4_count()

        prompt = prompt.replace("'", "").replace('"', '').replace("‘", "").replace("’", "").replace("“", "").replace("”", "")
        # Generate and run script
        script = generate_script(prompt, SEED)
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        print(f"Generating video with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new MP4
        if not wait_for_new_mp4(initial_count):
            print("Timeout waiting for new MP4 output.")
        
        # Update seed
        SEED += 1
        
        print(f"Finished processing prompt {idx + 1}\n")

if __name__ == "__main__":
    main()
