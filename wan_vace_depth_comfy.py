#### Wan VACE glass fruit 

https://www.reddit.com/r/StableDiffusion/comments/1ljljv1/wan_21_vace_makes_the_cut/
https://drive.google.com/drive/folders/1_3ONuuX5NxxyeoCWZruTgcWzsMTmGB_Z

https://huggingface.co/Kijai/WanVideo_comfy/tree/main

wget https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main/Wan2.1_14B_VACE-Q8_0.gguf

cp Wan2.1_14B_VACE-Q8_0.gguf ComfyUI/models/unet

wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
wget https://huggingface.co/alibaba-pai/Wan2.1-Fun-Reward-LoRAs/resolve/main/Wan2.1-Fun-14B-InP-MPS.safetensors
wget https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors
wget "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/Wan14B_RealismBoost.safetensors"
wget "https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/resolve/main/OtherLoRa's/DetailEnhancerV1.safetensors"

cp Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors ComfyUI/models/loras
cp Wan2.1-Fun-14B-InP-MPS.safetensors ComfyUI/models/loras
cp Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors ComfyUI/models/loras
cp Wan14B_RealismBoost.safetensors ComfyUI/models/loras
cp DetailEnhancerV1.safetensors ComfyUI/models/loras


An apple-shaped sculpture of translucent blue glass cleanly splits under a razor-sharp blade, its cross-section revealing pure glass continuity throughout.

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    image, frame_count, audio, _ = VHSLoadVideo('cut_fruit.mp4', 16, 1280, 720, 113, 0, 1, None, None, 'Wan')
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('An apple-shaped sculpture of translucent blue glass cleanly splits under a razor-sharp blade, its cross-section revealing pure glass continuity throughout.', clip)
    conditioning2 = CLIPTextEncode('subtitles, overall gray, worst quality, low quality, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, cluttered background, three legs, many people in the background, walking backwards', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    da_v2_model = DownloadAndLoadDepthAnythingV2Model('depth_anything_v2_vitl_fp32.safetensors')
    image2, _ = LoadImage('cut_wood.png')
    image2, width, height = ImageResizeKJv2(image2, 1088, 1088, 'lanczos', 'resize', '0, 0, 0', 'center', 32, 'cpu')
    image = ImageScale(image, 'lanczos', width, height, 'center')
    image = DepthAnythingV2(da_v2_model, image)
    positive, negative, latent, frame_count = WanVaceToVideo(conditioning, conditioning2, vae, 720, 720, frame_count, 1, 0.6900000000000002, image, None, image2)
    model = UnetLoaderGGUF('Wan2.1_14B_VACE-Q8_0.gguf')
    model = LoraLoaderModelOnly(model, 'Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors', 1.0000000000000002)
    model = LoraLoaderModelOnly(model, 'Wan2.1-Fun-14B-InP-MPS.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors', 1.0000000000000002)
    model = LoraLoaderModelOnly(model, 'Wan14B_RealismBoost.safetensors', 0.4000000000000001)
    model = LoraLoaderModelOnly(model, 'DetailEnhancerV1.safetensors', 0.4000000000000001)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    latent = KSampler(model, 80, 4, 1, 'lcm', 'sgm_uniform', positive, negative, latent, 1)
    latent = TrimVideoLatent(latent, frame_count)
    image3 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image3, 16, 0, 'Vace_loraStack_', 'video/h264-mp4', False, True, audio, None, None)
    SaveImage(image3, 'SLICED/SnozBerry')

vim run_depth_fruit.py

import os
import time
import subprocess
from pathlib import Path

# Configuration
SEED = 661695664686456
OUTPUT_DIR = 'ComfyUI/output/'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

glass_fruit_cuts = [
    "A strawberry-shaped sculpture of translucent crimson glass cleanly splits under a razor-sharp blade, its cross-section revealing refractive striations.",
    "A lemon-shaped sculpture of cloudy yellow glass cleanly splits under a razor-sharp blade, its cross-section revealing light-diffusing imperfections.",
    "A grape-shaped sculpture of deep violet glass cleanly splits under a razor-sharp blade, its cross-section showing concentric layers.",
    "A watermelon-shaped sculpture of dichroic green-red glass cleanly splits under a razor-sharp blade, its plane shifting colors as it moves.",
    "A peach-shaped sculpture of frosted orange glass cleanly splits under a razor-sharp blade, revealing a web of microscopic cracks.",
    "A blueberry-shaped sculpture of cobalt glass cleanly splits under a razor-sharp blade, exploding into precise geometric shards.",
    "A pomegranate-shaped sculpture of garnet-red glass cleanly splits under a razor-sharp blade, exposing seed-like glass bubbles.",
    "A lime-shaped sculpture of neon green glass cleanly splits under a razor-sharp blade, the fracture glowing faintly.",
    "A blackberry-shaped sculpture of obsidian glass cleanly splits under a razor-sharp blade, crumbling into tetrahedral fragments.",
    "A kiwi-shaped sculpture of fibrous green glass cleanly splits under a razor-sharp blade, revealing radial patterns.",
    "A mango-shaped sculpture of amber glass cleanly splits under a razor-sharp blade, showing swirling golden inclusions.",
    "A raspberry-shaped sculpture of rose glass cleanly splits under a razor-sharp blade, releasing prismatic reflections.",
    "A banana-shaped sculpture of canary glass cleanly splits under a razor-sharp blade, bending light along curved fractures.",
    "A plum-shaped sculpture of amethyst glass cleanly splits under a razor-sharp blade, exposing crystalline structures.",
    "A cherry-shaped sculpture of ruby glass cleanly splits under a razor-sharp blade, refracting light into crimson beams.",
    "A pineapple-shaped sculpture of honeycomb glass cleanly splits under a razor-sharp blade, revealing hexagonal chambers.",
    "A fig-shaped sculpture of violet-tinged glass cleanly splits under a razor-sharp blade, showing organic-looking fractures.",
    "A persimmon-shaped sculpture of pumpkin glass cleanly splits under a razor-sharp blade, with warm-toned refractions.",
    "A dragonfruit-shaped sculpture of speckled glass cleanly splits under a razor-sharp blade, exposing black seed voids.",
    "A starfruit-shaped sculpture of star-patterned glass cleanly splits under a razor-sharp blade, creating perfect stellar cross-sections."
]

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

def generate_script(prompt, seed):
    """Generate the ComfyUI script with the given prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    image, frame_count, audio, _ = VHSLoadVideo('cut_fruit.mp4', 16, 1280, 720, 113, 0, 1, None, None, 'Wan')
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    conditioning2 = CLIPTextEncode('subtitles, overall gray, worst quality, low quality, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, cluttered background, three legs, many people in the background, walking backwards', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    da_v2_model = DownloadAndLoadDepthAnythingV2Model('depth_anything_v2_vitl_fp32.safetensors')
    image2, _ = LoadImage('cut_wood.png')
    image2, width, height = ImageResizeKJv2(image2, 1088, 1088, 'lanczos', 'resize', '0, 0, 0', 'center', 32, 'cpu')
    image = ImageScale(image, 'lanczos', width, height, 'center')
    image = DepthAnythingV2(da_v2_model, image)
    positive, negative, latent, frame_count = WanVaceToVideo(conditioning, conditioning2, vae, 720, 720, frame_count, 1, 0.6900000000000002, image, None, image2)
    model = UnetLoaderGGUF('Wan2.1_14B_VACE-Q8_0.gguf')
    model = LoraLoaderModelOnly(model, 'Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors', 1.0000000000000002)
    model = LoraLoaderModelOnly(model, 'Wan2.1-Fun-14B-InP-MPS.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors', 1.0000000000000002)
    model = LoraLoaderModelOnly(model, 'Wan14B_RealismBoost.safetensors', 0.4000000000000001)
    model = LoraLoaderModelOnly(model, 'DetailEnhancerV1.safetensors', 0.4000000000000001)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    latent = KSampler(model, 80, 4, 1, 'lcm', 'sgm_uniform', positive, negative, latent, 1)
    latent = TrimVideoLatent(latent, frame_count)
    image3 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image3, 16, 0, 'Vace_loraStack_', 'video/h264-mp4', False, True, audio, None, None)
    SaveImage(image3, 'SLICED/SnozBerry')
"""
    return script_content

def main():
    SEED = 661695664686456
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each glass fruit description
    for idx, prompt in enumerate(glass_fruit_cuts):
        print(f"Processing item {idx + 1}/{len(glass_fruit_cuts)}")
        print(f"Prompt: {prompt[:100]}...")  # Print first 100 characters
        
        # Clean prompt text
        prompt = prompt.replace("'", "").replace('"', '').replace("‘", "").replace("’", "").replace("“", "").replace("”", "")
        
        # Generate and run script
        script = generate_script(prompt.replace("'", "\\'"), SEED)
        
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
        
        print(f"Finished processing item {idx + 1}\n")

if __name__ == "__main__":
    main()

#### Wan VACE anime relight V2V 

wget https://huggingface.co/svjack/Anime_Bright_Landscape_wan_2_1_14_B_text2video_lora/resolve/main/ani_bright_landscape_w14_outputs/ani_bright_landscape_w14_lora-step00005500.safetensors
wget https://huggingface.co/svjack/Wan_2_1_safetensors_pth/resolve/main/Wan2.1_T2V_14B_FusionX_LoRA.safetensors

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    image, frame_count, audio, _ = VHSLoadVideo('Flowers%20in%20spring___001.mp4', 16, 1280, 720, 113, 0, 1, None, None, 'Wan')
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('anime style ,The video presents a close-up view of a cluster of white flowers with five petals each, set against a backdrop of green leaves. The flowers are in full bloom, and their delicate petals are illuminated by natural light, giving them a soft glow. The leaves surrounding the flowers are a vibrant green, suggesting that the video was taken during the spring or summer season when plants are in full leaf. There is no movement or action within the frames; the focus remains on the stillness and beauty of the flowers.', clip)
    conditioning2 = CLIPTextEncode('subtitles, overall gray, worst quality, low quality, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, cluttered background, three legs, many people in the background, walking backwards', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    da_v2_model = DownloadAndLoadDepthAnythingV2Model('depth_anything_v2_vitl_fp32.safetensors')
    image2, _ = LoadImage('anime_light_tree_train.jpg')
    image2, width, height = ImageResizeKJv2(image2, 1088, 1088, 'lanczos', 'resize', '0, 0, 0', 'center', 32, 'cpu')
    image = ImageScale(image, 'lanczos', width, height, 'center')
    image = DepthAnythingV2(da_v2_model, image)
    positive, negative, latent, frame_count = WanVaceToVideo(conditioning, conditioning2, vae, 768, 512, frame_count, 1, 0.6900000000000002, image, None, image2)
    model = UnetLoaderGGUF('Wan2.1_14B_VACE-Q8_0.gguf')
    model = LoraLoaderModelOnly(model, 'Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'Wan2.1-Fun-14B-InP-MPS.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'ani_bright_landscape_w14_lora-step00005500.safetensors', 2.0000000000000004)
    model = LoraLoaderModelOnly(model, 'Wan2.1_T2V_14B_FusionX_LoRA.safetensors', 0.5000000000000001)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    latent = KSampler(model, 84, 4, 1, 'lcm', 'sgm_uniform', positive, negative, latent, 1)
    latent = TrimVideoLatent(latent, frame_count)
    image3 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image3, 16, 0, 'Vace_loraStack_', 'video/h264-mp4', False, True, audio, None, None)
    SaveImage(image3, 'SLICED/SnozBerry')


git clone https://huggingface.co/datasets/svjack/Scenery_Shot_Videos_Captioned

import os
import random
import shutil
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"无法获取视频时长 {video_path}: {e}")
        return None

def process_files(source_dir, target_dir, num_pairs=50, min_duration=5, max_duration=7):
    """处理文件"""
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    video_files = random.sample(video_files, len(video_files))
    
    # 筛选符合条件的视频（5-7秒）
    valid_pairs = []
    for video_file in tqdm(video_files):
        video_path = os.path.join(source_dir, video_file)
        duration = get_video_duration(video_path)
        
        if duration is not None and min_duration <= duration <= max_duration:
            txt_file = video_file.replace('.mp4', '.txt')
            txt_path = os.path.join(source_dir, txt_file)
            
            if os.path.exists(txt_path):
                valid_pairs.append((video_file, txt_file))
                print("valid length :", len(valid_pairs))
                if len(valid_pairs) > 100:
                    break
    
    # 随机挑选指定数量的对儿
    if len(valid_pairs) < num_pairs:
        print(f"警告：只有 {len(valid_pairs)} 个符合条件的视频-文本对，少于请求的 {num_pairs} 个")
        selected_pairs = valid_pairs
    else:
        selected_pairs = random.sample(valid_pairs, num_pairs)
    
    # 处理选中的对儿
    for video_file, txt_file in selected_pairs:
        # 拷贝视频文件
        src_video = os.path.join(source_dir, video_file)
        dst_video = os.path.join(target_dir, video_file)
        shutil.copy2(src_video, dst_video)
        
        # 处理并拷贝文本文件
        src_txt = os.path.join(source_dir, txt_file)
        dst_txt = os.path.join(target_dir, txt_file)
        
        # 读取并修改文本内容
        with open(src_txt, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换文本
        modified_content = content.replace("In the style of scenery", "anime style")
        
        # 写入新文件
        with open(dst_txt, 'w', encoding='utf-8') as f:
            f.write(modified_content)
    
    print(f"成功处理 {len(selected_pairs)} 个视频-文本对，已保存到 {target_dir}")

# 使用示例
source_directory = "Scenery_Shot_Videos_Captioned"
target_directory = "Selected_Anime_Style_Videos"
process_files(source_directory, target_directory)

vim run_anime_light_depth.py

import os
import time
import subprocess
from pathlib import Path
import shutil

# Configuration
SEED = 661695664686456
OUTPUT_DIR = 'ComfyUI/output/'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'
SOURCE_DIR = 'Selected_Anime_Style_Videos'

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

def generate_script(video_file, prompt, seed):
    """Generate the ComfyUI script with the given video and prompt"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    image, frame_count, audio, _ = VHSLoadVideo('{video_file}', 16, 1280, 720, 113, 0, 1, None, None, 'Wan')
    clip = CLIPLoader('umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'wan', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    conditioning2 = CLIPTextEncode('subtitles, overall gray, worst quality, low quality, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, cluttered background, three legs, many people in the background, walking backwards', clip)
    vae = VAELoader('wan_2.1_vae.safetensors')
    da_v2_model = DownloadAndLoadDepthAnythingV2Model('depth_anything_v2_vitl_fp32.safetensors')
    image2, _ = LoadImage('anime_light_tree_train.jpg')
    image2, width, height = ImageResizeKJv2(image2, 1088, 1088, 'lanczos', 'resize', '0, 0, 0', 'center', 32, 'cpu')
    image = ImageScale(image, 'lanczos', width, height, 'center')
    image = DepthAnythingV2(da_v2_model, image)
    positive, negative, latent, frame_count = WanVaceToVideo(conditioning, conditioning2, vae, 768, 512, frame_count, 1, 0.6900000000000002, image, None, image2)
    model = UnetLoaderGGUF('Wan2.1_14B_VACE-Q8_0.gguf')
    model = LoraLoaderModelOnly(model, 'Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'Wan2.1-Fun-14B-InP-MPS.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'Wan21_AccVid_T2V_14B_lora_rank32_fp16.safetensors', 0.5000000000000001)
    model = LoraLoaderModelOnly(model, 'ani_bright_landscape_w14_lora-step00005500.safetensors', 3.0000000000000004)
    model = LoraLoaderModelOnly(model, 'Wan2.1_T2V_14B_FusionX_LoRA.safetensors', 0.5000000000000001)
    model = ModelSamplingSD3(model, 1.0000000000000002)
    latent = KSampler(model, 84, 4, 1, 'lcm', 'sgm_uniform', positive, negative, latent, 1)
    latent = TrimVideoLatent(latent, frame_count)
    image3 = VAEDecode(latent, vae)
    _ = VHSVideoCombine(image3, 16, 0, 'Vace_loraStack_', 'video/h264-mp4', False, True, audio, None, None)
    SaveImage(image3, 'SLICED/SnozBerry')
"""
    return script_content

def clean_text(text):
    """Clean text for use in script"""
    # Remove problematic characters
    for char in ["'", '"', "‘", "’", "“", "”"]:
        text = text.replace(char, "")
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    # Remove leading/trailing whitespace
    return text.strip()

def prepare_input_files():
    """Copy video files to input directory and prepare text prompts"""
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all video files in source directory
    video_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.mp4')])
    
    # Prepare list of (video_file, prompt) pairs
    video_prompt_pairs = []
    
    for video_file in video_files:
        # Get corresponding text file
        txt_file = os.path.splitext(video_file)[0] + '.txt'
        txt_path = os.path.join(SOURCE_DIR, txt_file)
        
        if os.path.exists(txt_path):
            # Read text file content
            with open(txt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
            
            # Clean the prompt text
            prompt = clean_text(prompt)
            
            # Copy video to input directory
            src_video = os.path.join(SOURCE_DIR, video_file)
            dst_video = os.path.join(INPUT_DIR, video_file)
            shutil.copy2(src_video, dst_video)
            
            video_prompt_pairs.append((video_file, prompt))
    
    return video_prompt_pairs

def main():
    SEED = 661695664686456
    # Prepare input files and get video-prompt pairs
    video_prompt_pairs = prepare_input_files()
    
    if not video_prompt_pairs:
        print("No valid video-text pairs found in source directory.")
        return
    
    print(f"Found {len(video_prompt_pairs)} video-text pairs to process.")
    
    # Process each video-prompt pair
    for idx, (video_file, prompt) in enumerate(video_prompt_pairs):
        print(f"\nProcessing pair {idx + 1}/{len(video_prompt_pairs)}")
        print(f"Video: {video_file}")
        print(f"Prompt: {prompt[:100]}...")  # Print first 100 characters
        
        # Generate and run script
        script = generate_script(video_file, prompt.replace("'", "\\'"), SEED)
        
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
        
        print(f"Finished processing pair {idx + 1}\n")

if __name__ == "__main__":
    main()
