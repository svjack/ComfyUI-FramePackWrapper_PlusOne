# https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX

# T2V
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

# I2V
# https://civitai.com/models/1678575?modelVersionId=1900322
# https://civitai.com/models/1681541?modelVersionId=1903275

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'Wan2.1_I2V_14B_FusionX_LoRA.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('The poster prominently displays the title "ELECTRONIC PULSE" in a sleek, metallic silver font with subtle blue neon glow, positioned boldly across the top third, accompanied by the tagline "Where Sound Meets Innovation" in minimalist white typography at the bottom. Behind the text, a young Asian male singer with fair, luminous skin and tousled jet-black hair takes center stage. He wears a futuristic chrome headset microphone curving around his jawline, wires seamlessly blending into a geometric backdrop of holographic circuit boards and floating synth modules. His eyes are closed in passionate performance, catching dynamic stage lights—electric blues and violets—that streak across the scene. Glowing particles of digital energy swirl around his outstretched hand, merging organic artistry with technological spectacle against a deep indigo void scattered with pixel constellations.', clip)
    conditioning2 = CLIPTextEncode('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    clip_vision = CLIPVisionLoader('clip_vision_h.safetensors')
    image, _ = LoadImage('image (67).jpg')
    clip_vision_output = CLIPVisionEncode(clip_vision, image, 'none')
    positive, negative, latent = WanImageToVideo(conditioning, conditioning2, vae, 768, 1024, 81, 1, clip_vision_output, image)
    latent = KSampler(model, 978254506507205, 8, 1, 'euler', 'beta', positive, negative, latent, 1)
    image2 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image2, 16, 0, 'FusionXi2v/FusionX', 'video/h264-mp4', False, True, None, None, None)

import os
import time
import subprocess
from pathlib import Path
from datasets import load_dataset

# Configuration
SEED = 661695664686456
OUTPUT_DIR = 'ComfyUI/output/FusionXi2v'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
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
    """Download images from Hugging Face dataset"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # 加载数据集
    dataset = load_dataset("svjack/InfiniteYou_PosterCraft_Wang_Leehom_Poster_FP8_Wang_WAV_text_mask_inpaint")
    
    # 下载图片和保存prompt
    image_files = []
    for i, item in enumerate(dataset['train']):
        image = item['Wang_Leehom_poster_image']
        image_path = os.path.join(INPUT_DIR, f"input_{i}.jpg")
        image.save(image_path)
        image_files.append((f"input_{i}.jpg", item['final_prompt']))
    
    return image_files

def generate_script(image_name, prompt, seed):
    """Generate the ComfyUI script with the given image and prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'Wan2.1_I2V_14B_FusionX_LoRA.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    conditioning2 = CLIPTextEncode('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    clip_vision = CLIPVisionLoader('clip_vision_h.safetensors')
    image, _ = LoadImage('{image_name}')
    clip_vision_output = CLIPVisionEncode(clip_vision, image, 'none')
    positive, negative, latent = WanImageToVideo(conditioning, conditioning2, vae, 768, 1024, 81, 1, clip_vision_output, image)
    latent = KSampler(model, {seed}, 8, 1, 'euler', 'beta', positive, negative, latent, 1)
    image2 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image2, 16, 0, 'FusionXi2v/FusionX', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def main():
    SEED = 661695664686456
    # 下载数据集图片和获取prompt
    print("Downloading images from Hugging Face dataset...")
    image_prompt_pairs = download_dataset_images()
    total_items = len(image_prompt_pairs)
    print(f"Downloaded {total_items} images with prompts.")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 循环处理每张图片
    for idx, (image_file, prompt) in enumerate(image_prompt_pairs):
        print(f"Processing item {idx + 1}/{total_items}: {image_file}")
        print(f"Prompt: {prompt[:100]}...")  # 打印前100个字符
        
        # 构建工作流脚本
        prompt = prompt.replace("'", "").replace('"', '').replace("‘", "").replace("’", "").replace("“", "").replace("”", "")
        # Generate and run script
        script = generate_script(image_file, prompt.replace("'", "\\'"), SEED)
        
        # 写入文件
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # 获取当前输出数量
        initial_count = get_latest_output_count()
        
        # 运行脚本
        print(f"Generating video with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # 等待新输出
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
        
        # 更新种子
        SEED -= 1
        
        print(f"Finished processing {image_file}\n")

if __name__ == "__main__":
    main()

### I2V
### dataset: https://huggingface.co/datasets/svjack/Image2Any_Start_Frame
### model: https://huggingface.co/Alibaba-Research-Intelligence-Computing/wan-toy-transform/resolve/main/adapter_model.safetensors

import os 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import subprocess
from pathlib import Path
from datasets import load_dataset

# Configuration
SEED = 661695664686456
OUTPUT_DIR = 'ComfyUI/output/FusionXi2v'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
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
    """Download images from Hugging Face dataset"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # 加载数据集
    dataset = load_dataset("svjack/Image2Any_Start_Frame")
    
    # 下载图片和保存prompt
    image_files = []
    for i, item in enumerate(dataset['train']):
        image = item['image']
        image_path = os.path.join(INPUT_DIR, f"input_{i}.jpg")
        image.save(image_path)
        image_files.append((f"input_{i}.jpg", item['en_name']))
    
    return image_files

def generate_script(image_name, prompt, seed):
    """Generate the ComfyUI script with the given image and prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'Wan2.1_I2V_14B_FusionX_LoRA.safetensors', 1)
    model = LoraLoaderModelOnly(model, 'adapter_model.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    conditioning2 = CLIPTextEncode('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    clip_vision = CLIPVisionLoader('clip_vision_h.safetensors')
    image, _ = LoadImage('{image_name}')
    clip_vision_output = CLIPVisionEncode(clip_vision, image, 'none')
    positive, negative, latent = WanImageToVideo(conditioning, conditioning2, vae, 1024, 768, 81, 1, clip_vision_output, image)
    latent = KSampler(model, {seed}, 12, 1, 'euler', 'beta', positive, negative, latent, 1)
    image2 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image2, 16, 0, 'FusionXi2v/FusionX', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def main():
    SEED = 661695664686456
    # 下载数据集图片和获取prompt
    print("Downloading images from Hugging Face dataset...")
    image_prompt_pairs = download_dataset_images()
    total_items = len(image_prompt_pairs)
    print(f"Downloaded {total_items} images with prompts.")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 循环处理每张图片
    for idx, (image_file, prompt) in enumerate(image_prompt_pairs):
        print(f"Processing item {idx + 1}/{total_items}: {image_file}")
        print(f"Prompt: {prompt[:100]}...")  # 打印前100个字符
        
        # 构建工作流脚本
        prompt = prompt.replace("'", "").replace('"', '').replace("‘", "").replace("’", "").replace("“", "").replace("”", "")
        # Generate and run script
        prompt = "The video opens with a clear view of a {}. Then it transforms to a b6e9636 JellyCat-style {}. It has a face and a cute, fluffy and playful appearance.".format(prompt, prompt)
        script = generate_script(image_file, prompt.replace("'", "\\'"), SEED)
        
        # 写入文件
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # 获取当前输出数量
        initial_count = get_latest_output_count()
        
        # 运行脚本
        print(f"Generating video with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # 等待新输出
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
        
        # 更新种子
        SEED -= 1
        
        print(f"Finished processing {image_file}\n")

if __name__ == "__main__":
    main()


#### model https://huggingface.co/svjack/Wan_2_1_safetensors_pth/resolve/main/Phantom_Wan_14B_FusionX_LoRA.safetensors
#### dataset https://huggingface.co/datasets/svjack/Image2Any_Armed_Police

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Phantom-Wan-14B_fp16.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'Phantom_Wan_14B_FusionX_LoRA.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 3.0000000000000004)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('''At the nuclear power plant open day, solemnity and whimsy achieve perfect harmony. On the left, CNNC's educational display features a textbook-perfect mushroom cloud simulation - its fluffy crown blooming against azure sky, the white stem tapering with scientific precision, accented by yellow '蘑菇云' text and corporate logos. To the right, an armed police mascot sits at attention in the interactive zone: his dark green uniform's star-and-laurel cap badge, red epaulets with golden stars, and 'People's Police' armband glinting under sunlight. With one hand in pocket and oversized earnest eyes, this disciplined yet approachable figure reassures visiting children. The primal energy of nuclear science and protective symbolism of law enforcement engage in silent dialogue within this blue-and-yellow themed educational space.''', clip)
    conditioning2 = CLIPTextEncode('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    image, _ = LoadImage('PLA.jpg')
    image2, _ = LoadImage('image (68).jpg')
    image3 = ImageBatch(image, image2)
    positive, negative_text, negative_img_text, latent = WanPhantomSubjectToVideo(conditioning, conditioning2, vae, 1024, 576, 121, 1, image3)
    conditioning3 = ConditioningCombine(negative_text, negative_img_text)
    latent = KSampler(model, 1049199754871590, 8, 1, 'uni_pc', 'simple', positive, conditioning3, latent, 1)
    image4 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image4, 24, 0, 'FusionXi2v/FusionX', 'video/h264-mp4', False, True, None, None, None)

vim run_phantom_batch.py

import os
import time
import subprocess
from pathlib import Path
from datasets import load_dataset

# Configuration
SEED = 661695664686456
OUTPUT_DIR = 'ComfyUI/output/FusionXi2v'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

def get_latest_output_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new MP4 file appears in the output directory"""
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
    """Download images from Hugging Face dataset"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("svjack/Image2Any_Armed_Police")
    
    # Download images and save prompts
    image_files = []
    for i, item in enumerate(dataset['train']):
        image = item['image']
        image_path = os.path.join(INPUT_DIR, f"input_{i}.jpg")
        image.save(image_path)
        image_files.append((f"input_{i}.jpg", item['english_prompt']))
    
    return image_files

def generate_script(image_name, prompt, seed):
    """Generate the ComfyUI script with the given image and prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('Phantom-Wan-14B_fp16.safetensors', 'default')
    model = LoraLoaderModelOnly(model, 'Phantom_Wan_14B_FusionX_LoRA.safetensors', 1)
    model = PathchSageAttentionKJ(model, 'auto')
    model = ModelPatchTorchSettings(model, True)
    model = ModelSamplingSD3(model, 3.0000000000000004)
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    conditioning2 = CLIPTextEncode('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    image, _ = LoadImage('PLA.jpg')
    image2, _ = LoadImage('{image_name}')
    image3 = ImageBatch(image, image2)
    positive, negative_text, negative_img_text, latent = WanPhantomSubjectToVideo(conditioning, conditioning2, vae, 1024, 576, 121, 1, image3)
    conditioning3 = ConditioningCombine(negative_text, negative_img_text)
    latent = KSampler(model, {seed}, 8, 1, 'uni_pc', 'simple', positive, conditioning3, latent, 1)
    image4 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image4, 24, 0, 'FusionXi2v/FusionX', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def main():
    SEED = 661695664686456
    # Download dataset images and get prompts
    print("Downloading images from Hugging Face dataset...")
    image_prompt_pairs = download_dataset_images()
    total_items = len(image_prompt_pairs)
    print(f"Downloaded {total_items} images with prompts.")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each image
    for idx, (image_file, prompt) in enumerate(image_prompt_pairs):
        print(f"Processing item {idx + 1}/{total_items}: {image_file}")
        print(f"Prompt: {prompt[:100]}...")  # Print first 100 characters
        
        # Clean prompt text
        prompt = prompt.replace("'", "").replace('"', '').replace("‘", "").replace("’", "").replace("“", "").replace("”", "")
        
        # Generate and run script
        script = generate_script(image_file, prompt.replace("'", "\\'"), SEED)
        
        # Write to file
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # Get current output count
        initial_count = get_latest_output_count()
        
        # Run script
        print(f"Generating video with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new output
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
        
        # Update seed
        SEED -= 1
        
        print(f"Finished processing {image_file}\n")

if __name__ == "__main__":
    main()
