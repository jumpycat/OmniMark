import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn


class WatermarkExtractor:    
    def __init__(
        self,
        checkpoint_path: str,
        bits: int = 48,
        device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.bits = bits
        
        self.wm_decoder = self._load_decoder(checkpoint_path)
        
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])
        ])
    
    def _load_decoder(self, checkpoint_path: str):
        print(f'Loading watermark decoder from {checkpoint_path}...')
        
        wm_decoder = EfficientNet.from_pretrained('efficientnet-b0')
        feature = wm_decoder._fc.in_features
        wm_decoder._fc = nn.Linear(
            in_features=feature,
            out_features=self.bits,
            bias=True
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        wm_decoder.load_state_dict(checkpoint['wm_decoder'], strict=True)
        
        wm_decoder.to(self.device)
        wm_decoder.eval()
        
        print('Watermark decoder loaded successfully!')
        return wm_decoder
    
    def extract_watermark(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            decoded = self.wm_decoder(img_tensor)
            watermark_pred = (decoded > 0.0).float().cpu().numpy()[0]
        
        return watermark_pred
    
    def extract_from_directory(self, image_dir: str):
        watermarks = []
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        print(f'Found {len(image_files)} images in {image_dir}')
        
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_dir, img_file)
            watermark = self.extract_watermark(img_path)
            watermarks.append(watermark)
            print(f'Processed {idx}/{len(image_files)}: {img_file}')
        
        return np.array(watermarks)
    
    def calculate_accuracy(
        self,
        extracted_watermarks: np.ndarray,
        ground_truth_watermark: np.ndarray
    ):

        if ground_truth_watermark.ndim == 1:
            ground_truth_watermark = ground_truth_watermark.reshape(1, -1)
        
        diff = np.abs(extracted_watermarks - ground_truth_watermark)
        bitwise_accs = 100 * (1.0 - np.mean(diff, axis=1))

        avg_accuracy = np.mean(bitwise_accs)
        
        return avg_accuracy, bitwise_accs
    
    def evaluate_directory(
        self,
        image_dir: str,
        ground_truth_watermark: np.ndarray,
        verbose: bool = True
    ):

        extracted_watermarks = self.extract_from_directory(image_dir)
        
        avg_accuracy, bitwise_accs = self.calculate_accuracy(
            extracted_watermarks,
            ground_truth_watermark
        )
        
        if verbose:
            print('\n' + '='*60)
            print('Watermark Extraction Results:')
            print('='*60)
            print(f'Total images: {len(bitwise_accs)}')
            print(f'Average accuracy: {avg_accuracy:.2f}%')
            print(f'Min accuracy: {np.min(bitwise_accs):.2f}%')
            print(f'Max accuracy: {np.max(bitwise_accs):.2f}%')
            print(f'Std accuracy: {np.std(bitwise_accs):.2f}%')
            print('='*60)
            
            if len(bitwise_accs) <= 20:
                print('\nPer-image accuracy:')
                for idx, acc in enumerate(bitwise_accs, 1):
                    print(f'  Image {idx}: {acc:.2f}%')
        
        return avg_accuracy, bitwise_accs


def main():
    
    CHECKPOINT_PATH = "checkpoint.pth"
    IMAGE_DIR = "outputs"
    BITS = 48
    
    torch.manual_seed(4)
    ground_truth_watermark = torch.zeros((1, BITS), dtype=torch.float).random_(0, 2).numpy()
    
    # ground_truth_watermark = np.load('watermark.npy')
    
    extractor = WatermarkExtractor(
        checkpoint_path=CHECKPOINT_PATH,
        bits=BITS,
        device='cuda'
    )
    
    avg_acc, bitwise_accs = extractor.evaluate_directory(
        image_dir=IMAGE_DIR,
        ground_truth_watermark=ground_truth_watermark,
        verbose=True
    )
    
    # single_image_path = os.path.join(IMAGE_DIR, '1.png')
    # watermark = extractor.extract_watermark(single_image_path)
    # acc = 100 * (1.0 - np.mean(np.abs(watermark - ground_truth_watermark[0])))
    # print(f'Single image accuracy: {acc:.2f}%')


if __name__ == '__main__':
    main()