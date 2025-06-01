# ComfyUI-FramePackWrapper_PlusOne

## Scaramouche Demo
- https://huggingface.co/tori29umai/FramePack_LoRA/blob/main/body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors

```python
vim run_hy.py

from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    # _ = FramePackTorchCompileSettings('inductor', False, 'default', False, 64, True, True)
    lora = FramePackLoraSelect('body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors', 1, False, None)
    model = LoadFramePackModel('FramePackI2V_HY_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, lora)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp16.safetensors', 'hunyuan_video', 'default')
    conditioning = CLIPTextEncode('Convert reference images of poses and expressions into character design images.', clip)
    conditioning2 = ConditioningZeroOut(conditioning)
    image, _ = LoadImage('image (5).webp')
    width, height = FramePackFindNearestBucket(image, 640)
    image2, _, _ = ImageResize(image, width, height, 'nearest', 'stretch', 'always', 0)
    vae = VAELoader('hunyuan_video_vae_bf16.safetensors')
    latent = VAEEncode(image2, vae)
    clip_vision = CLIPVisionLoader('sigclip_vision_patch14_384.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image2, 'none')
    image3, _ = LoadImage('Screenshot 2025-05-31 155854.png')
    image3, _, _ = ImageResize(image3, width, height, 'nearest', 'stretch', 'always', 0)
    latent2 = VAEEncode(image3, vae)
    clip_vision_output2 = CLIPVisionEncode(clip_vision, image3, 'none')
    samples = FramePackSingleFrameSampler(model, conditioning, conditioning2, latent, 30, True, 0.15, 1, 7.520000000000001, 0, 634936421976103, 9, 25, 'unipc_bh1', True, clip_vision_output, latent, 1, latent2, clip_vision_output2, 5, 10, None, None)
    image4 = VAEDecodeTiled(samples, vae, 256, 64, 64, 8)
    SaveImage(image4, 'ComfyUI')

vim run_hy_iter.py

import os
import time
import subprocess
from pathlib import Path
from datasets import load_dataset

# Configuration
SEED = 661695664686456
STYLE_IMAGE_PATH = 'Screenshot 2025-05-31 155854.png'  # 对应 Screenshot 2025-05-31 155854.png
OUTPUT_DIR = 'ComfyUI/output'
INPUT_DIR = 'ComfyUI/input'
PYTHON_PATH = '/environment/miniconda3/bin/python'

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

def download_sketch_images():
    """Download sketch images from Hugging Face dataset"""
    # 创建输入目录如果不存在
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    # 加载数据集
    dataset = load_dataset("svjack/Genshin_Impact_Scaramouche_Images_1024x1024_with_sketch")
    
    # 下载 sketch_image 列的图片
    for i, item in enumerate(dataset['train']):
        sketch_image = item['sketch_image']
        image_path = os.path.join(INPUT_DIR, f"sketch_{i}.png")
        sketch_image.save(image_path)
    
    return [f"sketch_{i}.png" for i in range(len(dataset['train']))]

def generate_script(image_name, seed):
    """Generate the ComfyUI script with the given image and seed"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    # _ = FramePackTorchCompileSettings('inductor', False, 'default', False, 64, True, True)
    lora = FramePackLoraSelect('body2img_V7_kisekaeichi_dim4_1e-3_512_768-000140.safetensors', 1, False, None)
    model = LoadFramePackModel('FramePackI2V_HY_fp8_e4m3fn.safetensors', 'bf16', 'fp8_e4m3fn', 'offload_device', 'sdpa', None, lora)
    clip = DualCLIPLoader('clip_l.safetensors', 'llava_llama3_fp16.safetensors', 'hunyuan_video', 'default')
    conditioning = CLIPTextEncode('Convert reference images of poses and expressions into character design images.', clip)
    conditioning2 = ConditioningZeroOut(conditioning)
    image, _ = LoadImage('{image_name}')
    width, height = FramePackFindNearestBucket(image, 640)
    image2, _, _ = ImageResize(image, width, height, 'nearest', 'stretch', 'always', 0)
    vae = VAELoader('hunyuan_video_vae_bf16.safetensors')
    latent = VAEEncode(image2, vae)
    clip_vision = CLIPVisionLoader('sigclip_vision_patch14_384.safetensors')
    clip_vision_output = CLIPVisionEncode(clip_vision, image2, 'none')
    image3, _ = LoadImage('{STYLE_IMAGE_PATH}')
    image3, _, _ = ImageResize(image3, width, height, 'nearest', 'stretch', 'always', 0)
    latent2 = VAEEncode(image3, vae)
    clip_vision_output2 = CLIPVisionEncode(clip_vision, image3, 'none')
    samples = FramePackSingleFrameSampler(model, conditioning, conditioning2, latent, 30, True, 0.15, 1, 7.520000000000001, 0, {seed}, 9, 25, 'unipc_bh1', True, clip_vision_output, latent, 1, latent2, clip_vision_output2, 5, 10, None, None)
    image4 = VAEDecodeTiled(samples, vae, 256, 64, 64, 8)
    SaveImage(image4, 'ComfyUI')
"""
    return script_content

def main():
    SEED = 661695664686456
    # 下载所有 sketch 图片
    print("Downloading sketch images from Hugging Face dataset...")
    image_files = download_sketch_images()
    total_images = len(image_files)
    print(f"Downloaded {total_images} images.")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 循环处理每张图片
    for idx, image_file in enumerate(image_files):
        print(f"Processing image {idx + 1}/{total_images}: {image_file}")
        
        # 构建工作流脚本
        script = generate_script(image_file, SEED)
        
        # 写入文件
        with open('run_comfyui_workflow.py', 'w') as f:
            f.write(script)
        
        # 获取当前输出数量
        initial_count = get_latest_output_count()
        
        # 运行脚本
        print(f"Generating image with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_comfyui_workflow.py'])
        
        # 等待新输出
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output.")
        
        # 更新种子
        SEED -= 1
        
        print(f"Finished processing {image_file}\n")

if __name__ == "__main__":
    main()
```

This repository is derived from [ComfyUI-FramePackWrapper_Plus](https://github.com/ShmuelRonen/ComfyUI-FramePackWrapper_Plus/tree/main) and was created for my own use. I have little intention to maintain it. Please feel free to improve it, especially Framepack 1f-mc.

An improved wrapper for the FramePack project that allows the creation of videos of any length based on reference images and LoRAs with F1 sampler.

## Features

- **F1 Sampler Support**: Uses the improved F1 video generation method for higher quality and better temporal coherence
- **LoRA Integration**: Full support for HunyuanVideo LoRAs with proper weight handling and fusion options
- **Timestamped Prompts**: Create dynamic videos with changing prompts at specific timestamps
- **Flexible Input Options**: Works with both reference images and empty latents for complete creative control
- **Resolution Control**: Automatic bucket finding for optimal video dimensions
- **Blend Control**: Smooth transitions between different prompts at timestamps

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tori29umai0123/ComfyUI-FramePackWrapper_PlusOne.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the necessary model files and place them in your models folder:
- FramePackI2V_HY: [HuggingFace Link](https://huggingface.co/lllyasviel/FramePackI2V_HY)
- FramePack_F1_I2V_HY: [HuggingFace Link](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503)

## Model Files

### Main Model Options
- [FramePackI2V_HY_fp8_e4m3fn.safetensors](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_fp8_e4m3fn.safetensors) - Optimized fp8 version (smaller file size)
- [FramePackI2V_HY_bf16.safetensors](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_bf16.safetensors) - BF16 version (better quality)

### Required Components
- **CLIP Vision**: [sigclip_vision_384](https://huggingface.co/Comfy-Org/sigclip_vision_384/tree/main)
- **Text Encoder and VAE**: [HunyuanVideo_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files)

## Usage

### Basic Workflow

1. Load the HunyuanVideo model with your preferred settings
2. (Optional) Add LoRAs with the FramePackLoraSelect node
3. Prepare your input image or empty latent
4. Set up CLIP Vision encoding for image embeddings
5. Create timestamped prompts with FramePackTimestampedTextEncode
6. Generate your video with FramePackSampler (F1)

### Example Workflow

![image](https://github.com/user-attachments/assets/e7ba12b5-41ef-484b-a796-801b701628a5)

### Timestamped Prompts

Use the following format for timestamped prompts:
```
[0s: A beautiful landscape, mountains in the background]
[5s-10s: Camera pans to reveal a lake, reflections of clouds]
[10s: A boat appears on the horizon, sailing slowly]
```

- `[Xs: prompt]`: Starts at X seconds and continues until the next timestamp
- `[Xs-Ys: prompt]`: Active from X seconds to Y seconds

### LoRA Usage

1. Place your HunyuanVideo LoRAs in the `ComfyUI/models/loras` folder
2. Use the FramePackLoraSelect node to add them to your workflow
3. Adjust strength as desired (typically 0.5-1.2)
4. Set fuse_lora to false for flexibility or true for performance

## Node Reference

### FramePackSampler (F1)
The main generation node using the F1 sampling technique.

**Inputs:**
- `model`: The loaded FramePack model
- `positive_timed_data`: Timestamped positive prompts
- `negative`: Negative prompt conditioning
- `start_latent`: Initial latent for generation
- `start_image_embeds`: CLIP Vision embeddings for start image
- `end_latent`: (Optional) End latent for transitions
- `end_image_embeds`: (Optional) CLIP Vision embeddings for end image
- `initial_samples`: (Optional) For video-to-video generation
- Various sampling parameters (steps, CFG, guidance scale, etc.)

### FramePackTimestampedTextEncode
Encodes text prompts with timestamps for timed conditioning.

**Inputs:**
- `clip`: CLIP text model
- `text`: Text prompt with timestamps
- `negative_text`: Negative prompt
- `total_second_length`: Video duration in seconds
- `latent_window_size`: Processing window size
- `prompt_blend_sections`: Number of sections to blend prompts

### FramePackLoraSelect
Selects and configures LoRA models.

**Inputs:**
- `lora`: LoRA model selection
- `strength`: LoRA strength (0.0-2.0)
- `fuse_lora`: Whether to fuse the LoRA weights into the base model
- `prev_lora`: (Optional) For chaining multiple LoRAs

### LoadFramePackModel / DownloadAndLoadFramePackModel
Loads the FramePack model with various precision options.

## Advanced Tips

1. **Resolution Control**: Use the FramePackFindNearestBucket node to optimize dimensions
2. **Memory Management**: Adjust gpu_memory_preservation for large models
3. **Blending Prompts**: Set prompt_blend_sections > 0 for smooth transitions
4. **Multiple LoRAs**: Chain several LoRAs together for combined effects
5. **Empty Latents**: Use an Empty Latent Image node when starting from scratch

## Troubleshooting

- **CUDA Out of Memory**: Reduce resolution, decrease latent_window_size, or increase gpu_memory_preservation
- **LoRA Loading Issues**: Ensure LoRAs are in the correct format (safetensors)
- **Video Artifacts**: Try increasing steps or adjusting CFG/guidance_scale

## Acknowledgements

- Based on the original [ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper) by kijai
- Uses models from [FramePack](https://github.com/lllyasviel/Fooocus-FramePack) by lllyasviel
- Special thanks to the ComfyUI and Stable Diffusion communities

## License

[MIT License](LICENSE)

## Credits

This project is an extension of the original [ComfyUI-FramePackWrapper](https://github.com/kijai/ComfyUI-FramePackWrapper) created by kijai. The original wrapper provided the foundation for working with FramePack models in ComfyUI.

ComfyUI-FramePackWrapper_Plus builds upon that foundation by adding support for:
- F1 sampler for improved temporal coherence
- LoRA integration for customized generation
- Timestamped prompts for dynamic video creation
- Additional workflow improvements and optimizations

Special thanks to kijai for the original implementation that made this extension possible.
