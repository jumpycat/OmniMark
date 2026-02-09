import os
import torch
import numpy as np
from omegaconf import OmegaConf
from diffusers import StableDiffusionPipeline
import torch.nn as nn
from utils_model import load_model_from_config


class SDImageGenerator:
    def __init__(
        self,
        ldm_config_path: str,
        ldm_checkpoint_path: str,
        decoder_checkpoint_path: str,
        model_name: str = 'Manojb/stable-diffusion-2-base',
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        self.ldm_decoder = self._load_custom_decoder(
            ldm_config_path,
            ldm_checkpoint_path,
            decoder_checkpoint_path
        )
        
        self.pipe = self._load_pipeline()
    
    def _load_custom_decoder(
        self,
        config_path: str,
        ldm_ckpt_path: str,
        decoder_ckpt_path: str
    ):
        print(f'Loading LDM model from {ldm_ckpt_path}...')
        
        config = OmegaConf.load(config_path)
        ldm_ae = load_model_from_config(config, ldm_ckpt_path)
        
        ldm_decoder = ldm_ae.first_stage_model
        ldm_decoder.encoder = nn.Identity()
        ldm_decoder.quant_conv = nn.Identity()
        ldm_decoder.eval()
        
        print(f'Loading decoder checkpoint from {decoder_ckpt_path}...')
        state_dict = torch.load(decoder_ckpt_path, map_location=self.device)
        unexpected_keys = ldm_decoder.load_state_dict(
            state_dict['ldm_decoder'],
            strict=False
        )
        
        if unexpected_keys.missing_keys or unexpected_keys.unexpected_keys:
            print(f'Warning: Unexpected keys during loading: {unexpected_keys}')
        
        return ldm_decoder
    
    def _load_pipeline(self):
        print(f'Loading Stable Diffusion pipeline: {self.model_name}...')
        
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        return pipe
    
    def generate_images(
        self,
        prompts,
        output_dir: str,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
        watermark_dim: int = 48,
        num_images_per_prompt: int = 1
    ):

        os.makedirs(output_dir, exist_ok=True)
        
        self._set_seed(seed)
        
        watermark = torch.zeros((1, watermark_dim), dtype=torch.float)
        watermark.random_(0, 2)
        watermark = watermark.to(self.device)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image_index = 0
        with torch.no_grad():
            for idx, prompt in enumerate(prompts, 1):
                print(f'Generating images for prompt {idx}/{len(prompts)}...')
                
                images = self.pipe(
                    prompt,
                    negative_prompt=None,  
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    height=height,
                    width=width,
                    if_wm=True,
                    my_decoder=self.ldm_decoder.decode,
                    wm=watermark
                ).images
                
                for img in images:
                    image_index += 1
                    output_path = os.path.join(output_dir, f'{image_index}.png')
                    img.save(output_path, format='png')
                    print(f'Saved: {output_path}')
        
        print(f'Generation complete! Total images: {image_index}')
    
    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def main():
    LDM_CONFIG = "configs/stable-diffusion/v2-inference-v-stu.yaml"
    LDM_CHECKPOINT = "stable-diffusion-2/768-v-ema.ckpt"
    DECODER_CHECKPOINT = "checkpoint.pth"
    OUTPUT_DIR = "outputs"
    
    prompts = [
    'a delicate apple made of opal hung on branch.',
    ]
    
    generator = SDImageGenerator(
        ldm_config_path=LDM_CONFIG,
        ldm_checkpoint_path=LDM_CHECKPOINT,
        decoder_checkpoint_path=DECODER_CHECKPOINT
    )
    
    generator.generate_images(
        prompts=prompts,
        output_dir=OUTPUT_DIR,
        seed=4,
        height=512,
        width=512
    )


if __name__ == '__main__':
    main()