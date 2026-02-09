import argparse
import os
import sys
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import time
import utils
import utils_img
import utils_model
from torch import optim
from datetime import timedelta

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, default='CoCo/coco_train/train_2017')

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="v2-inference.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_stu_config", type=str, default="v2-inference_stu.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="stable-diffusion-2/768-v-ema.ckpt", help="Path to the checkpoint file for the LDM model") 

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=6, help="Batch size for training")
    aa("--img_size", type=int, default=512, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=10.0, help="Weight of the image loss in the total loss")
    aa("--lambda_w", type=float, default=1.0, help="Weight of the watermark loss in the total loss")
    aa("--lr", type=float, default=3e-4)
    aa("--steps", type=int, default=200000, help="Number of steps to train the model for")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000, help="Frequency of saving generated images (in steps)")
    aa("--save_model_freq", type=int, default=10000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--seed", type=int, default=42)
    aa("--bits", type=int, default=48)
    aa("--warm_steps", type=int, default=15000)
    aa("--cum_times", type=float, default=10)
    aa("--resume", type=str, default=None)

    group = parser.add_argument_group('DA parameters')
    aa("--if_atk", action="store_true")
    aa("--attack", type=str,default='gbcnhja')

    return parser 

def main(params):
    # Create directories if not exist.
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%y%m%d%H%M%S")

    exp_path = os.path.join('runs',dt_string)
    model_save_dir = os.path.join('runs',dt_string,'models')
    sample_dir = os.path.join('runs',dt_string,'samples')
    os.makedirs(exp_path,exist_ok=True)
    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(sample_dir,exist_ok=True)

    cmd = ' '.join(sys.argv)
    with open(os.path.join(exp_path,'args.txt'), 'a') as file:
        for arg in vars(params):
            file.write(f"{arg}: {getattr(params, arg)}\n")
        file.write(f"{cmd}\n")


    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    
    # Loads the data
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.RandomCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan])

    Minus112ZeroOne = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])
    from torchvision.datasets import ImageFolder
    train_dataset = ImageFolder(params.train_dir, transform=vqgan_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8)

    # Loads LDM auto-encoder models pretrained
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae  = utils_model.load_model_from_config(config, params.ldm_ckpt).first_stage_model  # LatentDiffusion -> AutoencoderKL
    ldm_ae.eval()
    ldm_ae.to(device)
    
    # Loads student
    config_stu = OmegaConf.load(f"{params.ldm_stu_config}")
    stu = utils_model.load_model_from_config(config_stu, params.ldm_ckpt).first_stage_model
    stu.encoder = nn.Identity()
    stu.quant_conv = nn.Identity()
    stu.to(device)
    stu.decoder.train()

    # Loads hidden decoder
    from efficientnet_pytorch import EfficientNet
    wm_decoder = EfficientNet.from_pretrained('efficientnet-b0')
    feature = wm_decoder._fc.in_features
    wm_decoder._fc = nn.Linear(in_features=feature, out_features=params.bits, bias=True)
    wm_decoder.to(device)
    wm_decoder.train()

    optimizer = optim.Adam(list(stu.parameters()) + list(wm_decoder.parameters()), lr=params.lr)

    start_step = 1
    try:
        weights = torch.load(params.resume)
        wm_decoder.load_state_dict(weights['wm_decoder'], strict=True)
        stu.load_state_dict(weights['ldm_decoder'],strict=False)
        print('Weights Loaded!')
        start_step = 90001

        optimizer_state_dict = torch.load(weights['optimizer'], strict=True)
        optimizer.load_state_dict(optimizer_state_dict)

    except:
        pass

    for param in [*ldm_ae.parameters()]:
        param.requires_grad = False
    for param in [*stu.parameters(), *wm_decoder.parameters()]:
        param.requires_grad = True


    # Create losses
    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    my_mse = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    loss_w_1 = lambda decoded, keys: F.binary_cross_entropy_with_logits(decoded * 1, keys, reduction='mean')
    loss_w_10 = lambda decoded, keys: F.binary_cross_entropy_with_logits(decoded * 10, keys, reduction='mean')


    from attack_methods_01.attack_initializer import attack_initializer #For augmentation
    train_aug01 = attack_initializer(params.attack, is_train = True, device = device)


    img_ret = []
    wm_ret = []
    train_iter = iter(train_loader)
    start_time = time.time()
    for total_steps in range(start_step, params.steps + 1):
        try:
            imgs = next(train_iter)[0].to(device)
        except:
            train_iter = iter(train_loader)
            imgs = next(train_iter)[0].to(device)
        watermark = torch.zeros((imgs.size(0), params.bits), dtype=torch.float).random_(0, 2).to(device)

        # encode images
        with torch.no_grad():
            imgs_z = ldm_ae.encode(imgs) 
            imgs_z = imgs_z.mode()

        imgs_marked = stu.decode(imgs_z, watermark) 

        if params.if_atk:
            imgs_marked_aug = Minus112ZeroOne(imgs_marked)
            imgs_marked_aug = train_aug01(imgs_marked_aug).clamp(0, 1)
            decoded = wm_decoder(imgs_marked_aug) 
        else:
            decoded = wm_decoder(Minus112ZeroOne(imgs_marked)) 

        # compute loss
        if total_steps <= params.warm_steps:
            lossw = loss_w_10(decoded, watermark)
            lossi = my_mse(imgs, imgs_marked)
            lambda_w = params.lambda_w          # 1
            lambda_i = params.lambda_i / 10     # 10 -> 1

        elif total_steps <= 6 * params.warm_steps:
            lossw = loss_w_1(decoded, watermark)
            lossi = loss_fn_vgg.forward(imgs, imgs_marked).mean()
            lambda_w = params.lambda_w          # 1
            lambda_i = params.lambda_i          # 10

        else:
            lossw = loss_w_1(decoded, watermark)
            lossi = loss_fn_vgg.forward(imgs, imgs_marked).mean()
            lambda_w = params.lambda_w / 10    
            lambda_i = params.lambda_i         



        loss = lambda_w * lossw + lambda_i * lossi
        loss.backward()

        if total_steps <= 2 * params.warm_steps:
            optimizer.step()
            optimizer.zero_grad()
        else:
            if total_steps % params.cum_times == 0:
                optimizer.step()
                optimizer.zero_grad()

        if total_steps % params.log_freq == 0:
            last_time = time.time() - start_time

            wm_predicted = (decoded > 0.0).float()
            bitwise_acc = 100 * (1.0 - torch.mean(torch.abs(watermark - wm_predicted)))

            log = f"{dt_string} {total_steps:06} | LPIPS: {lossi.item():.5f} | WM: {lossw.item():.5f} | Acc: {bitwise_acc.item():.2f} | Time:[{format_time(last_time)}]"
            print(log)
            with open(os.path.join(exp_path,'logs.txt'), 'a', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')
        
            img_ret.append(lossi.item())
            wm_ret.append(bitwise_acc.item())


        # save images during training
        if total_steps % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(sample_dir, f'{total_steps:06}_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_marked),0,1), os.path.join(sample_dir, f'{total_steps:06}_train_w.png'), nrow=8)
    
        if total_steps % params.save_model_freq == 0:
            save_dict = {
                'ldm_decoder': stu.state_dict(),
                'wm_decoder': wm_decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'params': params,
            }

            # Save checkpoint
            torch.save(save_dict, os.path.join(model_save_dir, f"checkpoint_{total_steps:06d}.pth"))

        if total_steps % params.save_img_freq == 0 and total_steps >= params.save_img_freq:
            log = f"{total_steps} Mean Img: {np.mean(img_ret[-100:]):.5f} Mean Acc: {np.mean(wm_ret[-100:]):.5f}"
            with open(os.path.join(exp_path,'results.txt'), 'a', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')

if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)