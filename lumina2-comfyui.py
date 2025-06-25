#### Lumina‑Image‑2.0 neta-art/Neta-Lumina

https://huggingface.co/neta-art/Neta-Lumina

https://comfyui-wiki.com/en/tutorial/advanced/lumina-image-2

https://civitai.com/models/1257597/art-style-of-studio-ghibli-lumina-image-20

python lumina2comfy.py "path/to/my/lora.safetensors"


wget https://huggingface.co/neta-art/Neta-Lumina/resolve/main/ae.safetensors
wget https://huggingface.co/neta-art/Neta-Lumina/resolve/main/gemma_2_2b_fp16.safetensors
wget https://huggingface.co/neta-art/Neta-Lumina/resolve/main/neta-lumina-beta-0624-aes.safetensors

wget https://huggingface.co/neta-art/NetaLumina_Alpha/resolve/main/NetaAniLumina_Alpha_full_roundnnnn_ep6_s127716.pth

huggingface-cli login
huggingface-cli download neta-art/NetaLumina_Alpha NetaAniLumina_Alpha_full_roundnnnn_ep6_s127716.pth --local-dir .

huggingface-cli download neta-art/NetaLumina_Alpha NetaAniLumina_Alpha_aes_round4_ep18_s28705.pth --local-dir .


cp NetaAniLumina_Alpha_full_roundnnnn_ep6_s127716.pth ComfyUI/models/unet/

cp NetaAniLumina_Alpha_aes_round4_ep18_s28705.pth ComfyUI/models/unet/


cp gemma_2_2b_fp16.safetensors ComfyUI/models/text_encoders/
cp ae.safetensors ComfyUI/models/vae/

I captioned them using JoyCaption Alpha Two (locally) in "descriptive/long" mode and prefixed each caption with the phrase "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> Studio Ghibli style."
