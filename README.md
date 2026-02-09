# OmniMark: Efficient and Scalable Latent Diffusion Model Fingerprinting

This repository contains the official code for the paper:
**"OmniMark: Efficient and Scalable Latent Diffusion Model Fingerprinting"**  
*(Jianwei Fei, Yunshu Dai, Zhihua Xia, Fangjun Huang, Jiantao Zhou, AAAI 2025)*

## 📦 Repository Overview
OmniMark is an efficient watermarking framework for Latent Diffusion Models (LDMs) that embeds imperceptible fingerprints into generated images. This repository provides training, generation, and extraction code.

## 🚀 Quick Start

**Download pretrained checkpoint:**
- 📥 [Google Drive Link](https://drive.google.com/file/d/1f47kMWH3jfcgwFKhvPGMbOIs_A9W21Fj/view?usp=drive_link)

After downloading, place `checkpoint.pth` in the project root directory.

**Additional requirements:**
- Stable Diffusion 2 base checkpoint: Download `768-v-ema.ckpt` from [Stability AI](https://huggingface.co/Manojb/stable-diffusion-2-base) and place in `stable-diffusion-2/` directory
- Configuration files: Ensure you have the required YAML config files in `configs/stable-diffusion/`

## 📋 Usage

### 1. Training

Train the watermark decoder and LDM decoder on your dataset:

```bash
python my_trainer.py 
```

**Key parameters:**
- `--train_dir`: Path to training images (ImageFolder format)
- `--bits`: Watermark length (default: 48 bits)
- `--steps`: Total training steps
- `--lambda_i`: Weight for image quality loss (LPIPS)
- `--lambda_w`: Weight for watermark loss (BCE)
- `--warm_steps`: Warmup steps for training schedule (default: 15000)
- `--resume`: Path to checkpoint for resuming training


### 2. Image Generation

Generate watermarked images using the trained model:

```bash
python gen_image.py
```

**Configuration in script:**
```python
# Edit gen_image.py to customize:
LDM_CONFIG = "configs/stable-diffusion/v2-inference-v-stu.yaml"
LDM_CHECKPOINT = "stable-diffusion-2/768-v-ema.ckpt"
DECODER_CHECKPOINT = "checkpoint.pth"  # Your downloaded checkpoint
OUTPUT_DIR = "outputs"

prompts = [
    'a delicate apple made of opal hung on branch.',
    'a beautiful sunset over the ocean',
    # Add your prompts here
]

# Generation parameters
seed = 4              # Random seed (must match extraction seed!)
height = 512          # Image height
width = 512           # Image width
watermark_dim = 48    # Watermark bits (default: 48)
```

**Important:** The `seed` parameter in generation must match the seed used in extraction to correctly verify the watermark.

### 3. Watermark Extraction

Extract and verify watermarks from generated images:

```bash
python extract_wm.py
```

**Configuration in script:**
```python
# Edit extract_wm.py to customize:
CHECKPOINT_PATH = "checkpoint.pth"
IMAGE_DIR = "outputs"  # Directory containing watermarked images
BITS = 48

# Generate the same ground truth watermark using matching seed
torch.manual_seed(4)  # Must match generation seed!
ground_truth_watermark = torch.zeros((1, BITS), dtype=torch.float).random_(0, 2).numpy()
```

**Extract from single image:**
```python
extractor = WatermarkExtractor(
    checkpoint_path="checkpoint.pth",
    bits=48,
    device='cuda'
)
watermark = extractor.extract_watermark('path/to/image.png')
```

## 📝 Notes

- **Seed consistency**: Ensure the same random seed is used for generation and extraction to correctly verify watermarks
- **GPU memory**: Training requires ~40GB VRAM with batch_size=6
- **Image format**: Supports PNG, JPG, JPEG for extraction
- **Checkpoint format**: The checkpoint contains both the LDM decoder and watermark decoder states

## 📧 Contact

For questions or collaboration inquiries, please contact: fei_jianwei@163.com

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{fei2025omnimark,
  title={OmniMark: Efficient and Scalable Latent Diffusion Model Fingerprinting},
  author={Fei, Jianwei and Dai, Yunshu and Xia, Zhihua and Huang, Fangjun and Zhou, Jiantao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={16},
  pages={16550--16558},
  year={2025}
}
```
