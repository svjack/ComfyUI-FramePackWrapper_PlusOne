#### relight model: https://huggingface.co/ippanorc/animetic_light/resolve/main/animetic_light.safetensors
#### relight workflow: 

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    prompt = '''
    'Here's an English description of the image based on your details:
    The artwork features an elegant female character from "Genshin Impact" with long, flowing blonde hair adorned by a black ribbon. Her striking blue eyes stand out against her fair complexion. She wears a sophisticated, high-collared outfit in dark teal with intricate gold embellishments, paired with sleek black gloves. In her hand, she delicately holds a small white flower, adding to her graceful demeanor.
    The background is a tranquil outdoor setting with lush green trees under a bright blue sky scattered with soft white clouds. Floating yellow petals enhance the dreamy, whimsical atmosphere. The vibrant colors and centered composition draw focus to the character, creating a lively and enchanting scene.
    '''

    # _ = FramePackTorchCompileSettings('inductor', False, 'default', False, 64, True, True)
    lora = FramePackLoraSelect('animetic_light.safetensors', 1, True, None)
    model = LoadFramePackModel('FramePackI2V_HY_bf16.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, lora)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp8_scaled.safetensors', 'hunyuan_video', 'default')
    conditioning = CLIPTextEncode(prompt, clip)
    conditioning2 = ConditioningZeroOut(conditioning)
    image, _ = LoadImage('image (21).png')
    width, height = FramePackFindNearestBucket(image, 1024)
    image2, _, _ = ImageResize(image, width, height, 'nearest', 'stretch', 'always', 0)
    vae = VAELoader('hunyuan_video_vae_bf16.safetensors')
    latent = VAEEncode(image2, vae)
    clip_vision = CLIPVisionLoader('sigclip_vision_patch14_384.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image2, 'none')
    samples = FramePackSingleFrameSampler(model, conditioning, conditioning2, latent, 15, True, 0.15, 1, 10, 0, 581005381480709, 9, 25, 'unipc_bh1', False, clip_vision_output, latent, 1, None, None, 5, 10, None, None)
    image3 = VAEDecodeTiled(samples, vae, 256, 64, 64, 8)
    SaveImage(image3, 'ComfyUI')
    # _ = GetLatentRangeFromBatch(None, 1, 1)
    #images = ImageConcatMulti(2, image2, image3, 'right', False)
    #SaveImage(images, 'ComfyUI')

from PIL import Image, ImageOps
import os
import time
import subprocess
from pathlib import Path
from datasets import load_dataset

# Configuration
SEED = 581005381480709
OUTPUT_DIR = 'ComfyUI/output'
INPUT_DIR = 'ComfyUI/input'
PROMPT_DIR = 'ComfyUI/prompts'  # New directory for saving prompts
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def add_white_border(image, left, top, right):
    """左上右三方向加白边"""
    if image.mode == 'RGBA':
        fill_color = (255, 255, 255, 255)  # RGBA白色
    else:
        fill_color = (255, 255, 255)  # RGB白色
    
    # 仅扩展左上右（底部=0）
    bordered = ImageOps.expand(image, 
                              border=(left, top, right, 0), 
                              fill=fill_color)
    return bordered

def get_latest_output_count():
    """Return the number of PNG files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.png')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new PNG file appears in the output directory"""
    timeout = 6000  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def download_dataset_images():
    """Download whitebg images from Hugging Face dataset"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(PROMPT_DIR, exist_ok=True)  # Create prompts directory
    
    # Load and filter dataset
    dataset = load_dataset("svjack/Genshin_Impact_Birthday_Art_Images_Captioned_RMBG")["train"]
    dataset = dataset.filter(lambda x: x["blue_ratio"] < 0.7)
    
    # Process and save images with white borders
    image_prompt_pairs = []
    for i, item in enumerate(dataset):
        whitebg_image = item['whitebg_image']
        prompt = item['prompt']
        
        # Add white borders
        bordered_image = add_white_border(whitebg_image, 256, 128, 256)
        
        # Save image
        image_path = os.path.join(INPUT_DIR, f"genshin_{i}.png")
        bordered_image.save(image_path)
        
        # Save prompt to file
        prompt_path = os.path.join(PROMPT_DIR, f"genshin_{i}.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        image_prompt_pairs.append((f"genshin_{i}.png", prompt))
    
    return image_prompt_pairs

def generate_script(image_name, prompt, seed):
    """Generate the ComfyUI script with the given image and prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    prompt = '''{prompt}'''
    # _ = FramePackTorchCompileSettings('inductor', False, 'default', False, 64, True, True)
    lora = FramePackLoraSelect('animetic_light.safetensors', 1, True, None)
    model = LoadFramePackModel('FramePackI2V_HY_bf16.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, lora)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp8_scaled.safetensors', 'hunyuan_video', 'default')
    conditioning = CLIPTextEncode(prompt, clip)
    conditioning2 = ConditioningZeroOut(conditioning)
    image, _ = LoadImage('{image_name}')
    width, height = FramePackFindNearestBucket(image, 1024)
    image2, _, _ = ImageResize(image, width, height, 'nearest', 'stretch', 'always', 0)
    vae = VAELoader('hunyuan_video_vae_bf16.safetensors')
    latent = VAEEncode(image2, vae)
    clip_vision = CLIPVisionLoader('sigclip_vision_patch14_384.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image2, 'none')
    samples = FramePackSingleFrameSampler(model, conditioning, conditioning2, latent, 15, True, 0.15, 1, 10, 0, {seed}, 9, 25, 'unipc_bh1', False, clip_vision_output, latent, 1, None, None, 5, 10, None, None)
    image3 = VAEDecodeTiled(samples, vae, 256, 64, 64, 8)
    SaveImage(image3, 'ComfyUI')
"""
    return script_content

def main():
    SEED = 581005381480709
    # Download and process dataset images
    print("Downloading and processing Genshin Impact birthday art images...")
    image_prompt_pairs = download_dataset_images()
    total_images = len(image_prompt_pairs)
    print(f"Processed {total_images} images with prompts.")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each image-prompt pair
    for idx, (image_file, prompt) in enumerate(image_prompt_pairs):
        print(f"Processing image {idx + 1}/{total_images}: {image_file}")
        print(f"Using prompt: {prompt[:100]}...")  # Print first 100 chars of prompt
        
        # Generate workflow script
        script = generate_script(image_file, prompt, SEED)
        
        # Write to file
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # Get current output count
        initial_count = get_latest_output_count()
        
        # Run script
        print(f"Generating image with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new output
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
        
        # Update seed
        SEED -= 1
        
        print(f"Finished processing {image_file}\n")

if __name__ == "__main__":
    main()
