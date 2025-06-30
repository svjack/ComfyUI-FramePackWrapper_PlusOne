#### install models 
#### kontext 5连拍生成图片
#### vace 5 连拍生成视频

#### install ffmpeg

#### deepseek 导演提示词
设计一个男性的五连拍的5个动作，主要集中在手和表情。
请严格遵守以下所有要求：
1. **数量**：必须正好生成五个姿势。
2. **多样性**：避免姿势之间的相似性。
3. **独立性**：只描述人物的动作，无需描述环境、人物衣服、已经外貌等其他特征。
4. **输出语言**：所有的姿势描述都必须使用英文。
5. **核心格式**：每一个姿势描述都必须以“make the mam”开头。
6. **最终输出格式**：用英文输出，每行1句，不要有额外输出

#### without face fix (kontext + vace)

vim run_multi.py

import os
import time
import shutil
from pathlib import Path
import subprocess

# Configuration
SEED = 757755269765134
OUTPUT_DIR = 'ComfyUI/output/'
INPUT_DIR = 'ComfyUI/input/'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'

current_image = 'xiang_idlo.jpg'

EDIT_PROMPTS = '''
make the man adjust his tie with one hand while giving a slight nod and a warm smile.  
make the man rub his chin thoughtfully with his thumb and index finger, eyes narrowed in concentration.  
make the man raise both hands palms-up in a shrug, eyebrows raised in playful confusion.  
make the man fold his arms tightly across his chest, jaw set with a determined expression.  
make the man snap his fingers with one hand while pointing forward with the other, mouth open as if making a sharp remark.
'''

EDIT_PROMPTS = list(map(lambda y: y.strip(), filter(lambda x: x.strip() ,EDIT_PROMPTS.split("\n"))))

# Edit prompts
#EDIT_PROMPTS = [
#    #"transform it into Ghibli style",
#    "make the man adjust his tie with one hand while giving a slight nod and a warm smile.",
#    "make the man rub his chin thoughtfully with his thumb and index finger, eyes narrowed in concentration.",
#    "make the man raise both hands palms-up in a shrug, eyebrows raised in playful confusion.",
#    "make the man fold his arms tightly across his chest, jaw set with a determined expression.",
#    "make the man snap his fingers with one hand while pointing forward with the other, mouth open as if making a sharp remark."
#]

def get_latest_output_count():
    """Return the number of PNG files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.png')))
    except:
        return 0

def get_latest_video_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count, is_video=False):
    """Wait until a new file appears in the output directory"""
    timeout = 600  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_count = get_latest_video_count() if is_video else get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def copy_latest_output_to_input():
    """Copy the latest output image to input directory"""
    # Get all PNG files in output directory sorted by modification time
    output_files = sorted(Path(OUTPUT_DIR).glob('*.png'), key=os.path.getmtime, reverse=True)
    if not output_files:
        return None
    
    latest_output = output_files[0]
    input_path = Path(INPUT_DIR) / latest_output.name
    
    # Copy the file
    shutil.copy2(latest_output, input_path)
    return input_path.name

def generate_kontext_script(seed, prompt, input_image):
    """Generate the ComfyUI script for image editing"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('flux1-kontext-dev-fp8-e4m3fn.safetensors', 'default')
    #model = LoraLoaderModelOnly(model, 'omni_con_flux_kontext_lora_v1_000005000.safetensors', 1)
    clip = DualCLIPLoader('clip_l.safetensors', 't5xxl_fp8_e4m3fn_scaled.safetensors', 'flux', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    image, _ = LoadImage('{input_image}')
    vae = VAELoader('ae.safetensors')
    latent = VAEEncode(image, vae)
    conditioning2 = ReferenceLatent(conditioning, latent)
    conditioning2 = FluxGuidance(conditioning2, 2.5)
    conditioning3 = ConditioningZeroOut(conditioning)
    latent2 = KSampler(model, {seed}, 20, 1, 'euler', 'simple', conditioning2, conditioning3, latent, 1)
    image2 = VAEDecode(latent2, vae)
    SaveImage(image2, 'ComfyUI')
    image3 = ImageStitch(image, 'right', True, 0, 'white', None)
    image3 = FluxKontextImageScale(image3)
    PreviewImage(image3)
"""
    return script_content

def generate_vace_script(image1, image2, prompt1, prompt2):
    """Generate the ComfyUI script for video generation"""
    combined_prompt = f"{prompt2}"
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    vae = WanVideoVAELoader('wan_2.1_vae.safetensors', 'bf16')
    block_swap_args = WanVideoBlockSwap(30, False, False, True, 15)
    lora = WanVideoLoraSelect('Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors', 1, None, None, False)
    vace_model = WanVideoVACEModelSelect('Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors')
    model = WanVideoModelLoader('Wan2_1-T2V-14B_fp8_e4m3fn.safetensors', 'fp16', 'fp8_e4m3fn', 'offload_device', 'sageattn', None, block_swap_args, lora, None, vace_model, None)
    value = INTConstant(480)
    value2 = INTConstant(832)
    value3 = INTConstant(49)
    image, _ = LoadImage('{image1}')
    image, _, _ = ImageResizeKJv2(image, value, value2, 'lanczos', 'crop', '0, 0, 0', 'center', 2, 'cpu')
    image2, _ = LoadImage('{image2}')
    image2, _, _ = ImageResizeKJv2(image2, value, value2, 'lanczos', 'crop', '0, 0, 0', 'center', 2, 'cpu')
    images, masks = WanVideoVACEStartToEndFrame(value3, 0.5, image, image2, None, None)
    vace_embeds = WanVideoVACEEncode(vae, value, value2, value3, 1.0000000000000002, 0, 1, images, image, masks, None, False)
    wan_t5_model = LoadWanVideoT5TextEncoder('umt5-xxl-enc-bf16.safetensors', 'bf16', 'offload_device', 'disabled')
    text_embeds = WanVideoTextEncode(wan_t5_model, '{combined_prompt}', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', True, None)
    slg_args = WanVideoSLG('9,10', 0.20000000000000004, 0.5000000000000001)
    exp_args = WanVideoExperimentalArgs('', True, False, 0, False, 1, 1.25, 20)
    samples = WanVideoSampler(model, vace_embeds, 8, 1.0000000000000002, 5.000000000000001, 815541497330794, True, 'lcm', 0, text_embeds, None, 1, None, None, None, None, '', slg_args, 'comfy', None, exp_args, None, None, None, None)
    images2 = WanVideoDecode(vae, samples, False, 272, 272, 144, 128)
    _ = VHSVideoCombine(images2, 16, 0, 'AnimateDiff', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def get_sorted_output_images():
    """Get all output images sorted by modification time"""
    return sorted(Path(OUTPUT_DIR).glob('*.png'), key=os.path.getmtime)

def main(current_image, SEED):
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    print("Starting image processing with edit prompts...")
    
    # Process the initial image through all edit prompts
    #current_image = 'xiang_idlo.jpg'
    
    for i, prompt in enumerate(EDIT_PROMPTS):
        print(f"\nProcessing edit {i + 1}/{len(EDIT_PROMPTS)}")
        print(f"Prompt: {prompt}")
        
        # Generate and run kontext script
        script = generate_kontext_script(SEED + i, prompt, current_image)
        
        # Write to file
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # Get current output count
        initial_count = get_latest_output_count()
        
        # Run script
        print(f"Processing image with seed: {SEED + i}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new output
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
            continue
        
        # Copy output to input for next iteration
        new_image = copy_latest_output_to_input()
        if new_image:
            current_image = new_image
            print(f"Saved output as {new_image} for next iteration")
    
    print("\nFinished processing all edit prompts. Starting video generation...")
    
    # Get all generated images sorted by time
    output_images = get_sorted_output_images()
    if len(output_images) < 2:
        print("Not enough images generated for video creation")
        return
    
    # Process images in pairs for video generation
    video_initial_count = get_latest_video_count()
    
    for i in range(len(output_images) - 1):
        print(f"\nGenerating video for image pair {i + 1}/{len(output_images) - 1}")
        
        # Get the two images and their prompts
        img1 = output_images[i].name
        img2 = output_images[i + 1].name
        prompt1 = EDIT_PROMPTS[i]
        prompt2 = EDIT_PROMPTS[i + 1]
        
        print(f"Using images: {img1} and {img2}")
        print(f"With prompts: {prompt1} and {prompt2}")
        
        # Generate and run vace script
        script = generate_vace_script(img1, img2, prompt1, prompt2)
        
        # Write to file
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # Run script
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new video output
        if not wait_for_new_output(video_initial_count, is_video=True):
            print("Timeout waiting for new video output.")
        
        video_initial_count += 1
    
    print("\nFinished all processing!")

if __name__ == "__main__":
    main(current_image, SEED)



#### with gan face fix (kontext + vace)

### Image Face restore

https://huggingface.co/spaces/svjack/Image_Face_Upscale_Restoration-GFPGAN

pip install gradio spaces "httpx[socks]"
pip install -r requirements.txt

vim /environment/miniconda3/lib/python3.11/site-packages/basicsr/data/degradations.py
from torchvision.transforms.functional_tensor import rgb_to_grayscale
->
from torchvision.transforms._functional_tensor import rgb_to_grayscale


from gradio_client import Client, handle_file

client = Client("http://localhost:7860")
result = client.predict(
		img=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		version="v1.4",
		scale=2,
		api_name="/inference"
)
print(result)

from PIL import Image
Image.open(result[-1])

### 

import os
import time
from pathlib import Path
import subprocess
from PIL import Image

# Configuration
SEED = 757755269765134
OUTPUT_DIR = 'ComfyUI/output/'
INPUT_DIR = 'ComfyUI/input/'
PYTHON_PATH = '/environment/miniconda3/envs/system/bin/python'
GFPGAN_SCRIPT = 'run_gfpgan_processing.py'

current_image = 'xiang_idlo.jpg'

EDIT_PROMPTS = '''
make the man adjust his tie with one hand while giving a slight nod and a warm smile.  
make the man rub his chin thoughtfully with his thumb and index finger, eyes narrowed in concentration.  
make the man raise both hands palms-up in a shrug, eyebrows raised in playful confusion.  
make the man fold his arms tightly across his chest, jaw set with a determined expression.  
make the man snap his fingers with one hand while pointing forward with the other, mouth open as if making a sharp remark.
'''

EDIT_PROMPTS = list(map(lambda y: y.strip(), filter(lambda x: x.strip(), EDIT_PROMPTS.split("\n"))))

def cleanup_directory(directory):
    """Remove all files in the specified directory"""
    for file in Path(directory).glob('*'):
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def copy_current_image_to_input():
    """Copy the current image to input directory and return its new path"""
    src = Path(current_image)
    dst = Path(INPUT_DIR) / src.name
    try:
        with open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
            f_dst.write(f_src.read())
        return dst.name
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return None

def concatenate_videos(output_path='final_output.mp4'):
    """Concatenate all MP4 files in output directory in alphabetical order"""
    video_files = sorted(Path(OUTPUT_DIR).glob('*.mp4'))
    if not video_files:
        print("No MP4 files found to concatenate")
        return
    
    # Create text file with list of videos for ffmpeg
    list_file = Path(OUTPUT_DIR) / 'video_list.txt'
    with open(list_file, 'w') as f:
        for video in video_files:
            f.write(f"file '{video.name}'\n")
    
    # Run ffmpeg to concatenate videos
    try:
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c', 'copy',
            str(Path(OUTPUT_DIR) / output_path)
        ]
        subprocess.run(cmd, check=True)
        print(f"Successfully created concatenated video at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating videos: {e}")
    finally:
        # Clean up the list file
        list_file.unlink(missing_ok=True)

def get_latest_output_count():
    """Return the number of PNG files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.png')))
    except:
        return 0

def get_latest_video_count():
    """Return the number of MP4 files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.mp4')))
    except:
        return 0

def wait_for_new_output(initial_count, is_video=False):
    """Wait until a new file appears in the output directory"""
    timeout = 600  # seconds
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        current_count = get_latest_video_count() if is_video else get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_gfpgan_script(input_path, output_path, version="v1.4", scale=2):
    """Generate the GFPGAN processing script"""
    script_content = f"""import os
import cv2
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import sys
sys.path.insert(0, "Image_Face_Upscale_Restoration-GFPGAN")
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

if not os.path.exists('realesr-general-x4v3.pth'):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")

if not os.path.exists('GFPGANv1.4.pth'):
    os.system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")

def main():
    # Initialize models
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    model_path = 'realesr-general-x4v3.pth'
    half = True if torch.cuda.is_available() else False
    upsampler = RealESRGANer(
        scale=4, 
        model_path=model_path, 
        model=model, 
        tile=0, 
        tile_pad=10, 
        pre_pad=0, 
        half=half
    )

    # Process image
    img = cv2.imread('{input_path}', cv2.IMREAD_UNCHANGED)
    original_height, original_width = img.shape[:2]
    
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    elif len(img.shape) == 2:
        img_mode = None
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_mode = None

    h, w = img.shape[0:2]
    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # Initialize face enhancer
    if '{version}' == 'v1.2':
        face_enhancer = GFPGANer(
            model_path='GFPGANv1.2.pth', 
            upscale=2, 
            arch='clean', 
            channel_multiplier=2, 
            bg_upsampler=upsampler
        )
    elif '{version}' == 'v1.3':
        face_enhancer = GFPGANer(
            model_path='GFPGANv1.3.pth', 
            upscale=2, 
            arch='clean', 
            channel_multiplier=2, 
            bg_upsampler=upsampler
        )
    elif '{version}' == 'v1.4':
        face_enhancer = GFPGANer(
            model_path='GFPGANv1.4.pth', 
            upscale=2, 
            arch='clean', 
            channel_multiplier=2, 
            bg_upsampler=upsampler
        )
    elif '{version}' == 'RestoreFormer':
        face_enhancer = GFPGANer(
            model_path='RestoreFormer.pth', 
            upscale=2, 
            arch='RestoreFormer', 
            channel_multiplier=2, 
            bg_upsampler=upsampler
        )
    elif '{version}' == 'CodeFormer':
        face_enhancer = GFPGANer(
            model_path='CodeFormer.pth', 
            upscale=2, 
            arch='CodeFormer', 
            channel_multiplier=2, 
            bg_upsampler=upsampler
        )

    # Enhance image
    _, _, output = face_enhancer.enhance(
        img, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True
    )

    # Rescale if needed
    if {scale} != 2:
        interpolation = cv2.INTER_AREA if {scale} < 2 else cv2.INTER_LANCZOS4
        h, w = img.shape[0:2]
        output = cv2.resize(output, (int(w * {scale} / 2), int(h * {scale} / 2)), interpolation=interpolation)

    # Resize output to match original input dimensions
    output = cv2.resize(output, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)

    # Save output
    extension = 'png' if img_mode == 'RGBA' else 'jpg'
    cv2.imwrite('{output_path}', output)

if __name__ == "__main__":
    main()
"""
    return script_content

def copy_latest_output_to_input():
    """Process the latest output image through GFPGAN and save to input directory"""
    # Get all PNG files in output directory sorted by modification time
    output_files = sorted(Path(OUTPUT_DIR).glob('*.png'), key=os.path.getmtime, reverse=True)
    if not output_files:
        return None
    
    latest_output = str(output_files[0])
    input_path = Path(INPUT_DIR) / output_files[0].name
    
    # Generate GFPGAN processing script
    script = generate_gfpgan_script(
        input_path=latest_output,
        output_path=str(input_path),
        version="v1.4",
        scale=2
    )
    
    # Write script to file
    with open(GFPGAN_SCRIPT, 'w') as f:
        f.write(script)
    
    # Run the script in a separate process
    subprocess.run([PYTHON_PATH, GFPGAN_SCRIPT])
    
    return input_path.name

def generate_kontext_script(seed, prompt, input_image):
    """Generate the ComfyUI script for image editing"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    model = UNETLoader('flux1-kontext-dev-fp8-e4m3fn.safetensors', 'default')
    #model = LoraLoaderModelOnly(model, 'omni_con_flux_kontext_lora_v1_000005000.safetensors', 1)
    clip = DualCLIPLoader('clip_l.safetensors', 't5xxl_fp8_e4m3fn_scaled.safetensors', 'flux', 'default')
    conditioning = CLIPTextEncode('{prompt}', clip)
    image, _ = LoadImage('{input_image}')
    vae = VAELoader('ae.safetensors')
    latent = VAEEncode(image, vae)
    conditioning2 = ReferenceLatent(conditioning, latent)
    conditioning2 = FluxGuidance(conditioning2, 2.5)
    conditioning3 = ConditioningZeroOut(conditioning)
    latent2 = KSampler(model, {seed}, 20, 1, 'euler', 'simple', conditioning2, conditioning3, latent, 1)
    image2 = VAEDecode(latent2, vae)
    SaveImage(image2, 'ComfyUI')
    image3 = ImageStitch(image, 'right', True, 0, 'white', None)
    image3 = FluxKontextImageScale(image3)
    PreviewImage(image3)
"""
    return script_content

def generate_vace_script(image1, image2, prompt1, prompt2):
    """Generate the ComfyUI script for video generation"""
    combined_prompt = f"{prompt2}"
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    vae = WanVideoVAELoader('wan_2.1_vae.safetensors', 'bf16')
    block_swap_args = WanVideoBlockSwap(30, False, False, True, 15)
    lora = WanVideoLoraSelect('Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors', 1, None, None, False)
    vace_model = WanVideoVACEModelSelect('Wan2_1-VACE_module_14B_fp8_e4m3fn.safetensors')
    model = WanVideoModelLoader('Wan2_1-T2V-14B_fp8_e4m3fn.safetensors', 'fp16', 'fp8_e4m3fn', 'offload_device', 'sageattn', None, block_swap_args, lora, None, vace_model, None)
    value = INTConstant(480)
    value2 = INTConstant(832)
    value3 = INTConstant(49)
    image, _ = LoadImage('{image1}')
    image, _, _ = ImageResizeKJv2(image, value, value2, 'lanczos', 'crop', '0, 0, 0', 'center', 2, 'cpu')
    image2, _ = LoadImage('{image2}')
    image2, _, _ = ImageResizeKJv2(image2, value, value2, 'lanczos', 'crop', '0, 0, 0', 'center', 2, 'cpu')
    images, masks = WanVideoVACEStartToEndFrame(value3, 0.5, image, image2, None, None)
    vace_embeds = WanVideoVACEEncode(vae, value, value2, value3, 1.0000000000000002, 0, 1, images, image, masks, None, False)
    wan_t5_model = LoadWanVideoT5TextEncoder('umt5-xxl-enc-bf16.safetensors', 'bf16', 'offload_device', 'disabled')
    text_embeds = WanVideoTextEncode(wan_t5_model, '{combined_prompt}', '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', True, None)
    slg_args = WanVideoSLG('9,10', 0.20000000000000004, 0.5000000000000001)
    exp_args = WanVideoExperimentalArgs('', True, False, 0, False, 1, 1.25, 20)
    samples = WanVideoSampler(model, vace_embeds, 8, 1.0000000000000002, 5.000000000000001, 815541497330794, True, 'lcm', 0, text_embeds, None, 1, None, None, None, None, '', slg_args, 'comfy', None, exp_args, None, None, None, None)
    images2 = WanVideoDecode(vae, samples, False, 272, 272, 144, 128)
    _ = VHSVideoCombine(images2, 16, 0, 'AnimateDiff', 'video/h264-mp4', False, True, None, None, None)
"""
    return script_content

def get_sorted_output_images():
    """Get all output images sorted by modification time"""
    return sorted(Path(OUTPUT_DIR).glob('*.png'), key=os.path.getmtime)

def main(current_image, SEED):
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # 1. Clean up directories and copy current image to input
    print("Cleaning up directories...")
    cleanup_directory(OUTPUT_DIR)
    cleanup_directory(INPUT_DIR)
    
    print(f"Copying {current_image} to input directory...")
    current_image = copy_current_image_to_input()
    if not current_image:
        print("Failed to copy current image to input directory")
        return
    
    print("Starting image processing with edit prompts...")
    
    for i, prompt in enumerate(EDIT_PROMPTS):
        print(f"\nProcessing edit {i + 1}/{len(EDIT_PROMPTS)}")
        print(f"Prompt: {prompt}")
        
        # Generate and run kontext script
        script = generate_kontext_script(SEED + i, prompt, current_image)
        
        # Write to file
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # Get current output count
        initial_count = get_latest_output_count()
        
        # Run script
        print(f"Processing image with seed: {SEED + i}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new output
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
            continue
        
        # Process output through API and save to input
        new_image = copy_latest_output_to_input()
        if new_image:
            current_image = new_image
            print(f"Saved output as {new_image} for next iteration")
    
    print("\nFinished processing all edit prompts. Starting video generation...")
    
    # Get all generated images sorted by time
    output_images = get_sorted_output_images()
    if len(output_images) < 2:
        print("Not enough images generated for video creation")
        return
        
    # Process images in pairs for video generation
    video_initial_count = get_latest_video_count()
    
    # 修改循环范围，确保不会超出EDIT_PROMPTS的范围
    for i in range(len(output_images) - 1):
        print(f"\nGenerating video for image pair {i + 1}/{len(output_images) - 1}")
        
        # Get the two images
        img1 = output_images[i].name
        img2 = output_images[i + 1].name
        
        # 确保prompt2不会超出范围
        prompt1 = EDIT_PROMPTS[i] if i < len(EDIT_PROMPTS) else ""
        prompt2 = EDIT_PROMPTS[i + 1] if (i + 1) < len(EDIT_PROMPTS) else ""
        
        print(f"Using images: {img1} and {img2}")
        print(f"With prompts: {prompt1} and {prompt2}")
        
        # Generate and run vace script
        script = generate_vace_script(img1, img2, prompt1, prompt2)
        
        # Write to file
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # Run script
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # Wait for new video output
        if not wait_for_new_output(video_initial_count, is_video=True):
            print("Timeout waiting for new video output.")
        
        video_initial_count += 1
    
    # 2. Concatenate all generated videos at the end
    print("\nConcatenating all generated videos...")
    concatenate_videos()
    
    print("\nFinished all processing!")

if __name__ == "__main__":
    main(current_image, SEED)