import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

import os

import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, path):
        self.source_latents = os.listdir(path)
        self.source_latents.sort(key=lambda x: int(x[:-3]))
        self.path = path


    def __len__(self):
        return len(self.source_latents)

    def __getitem__(self, idx):
        source_latent_path = os.path.join(self.path, self.source_latents[idx])


        return source_latent_path


def generate(args, g_ema, device, mean_latent, dataloader):
    os.makedirs('generate', exist_ok=True)
    with torch.no_grad():
        g_ema.eval()
        global_i = 0
        for source_latent_path in tqdm(dataloader):

            latent = torch.load(source_latent_path[0], map_location='cuda')
            latent = latent.unsqueeze(0)
            print(latent.shape)
            sample, _ = g_ema(
                [latent], truncation=args.truncation, truncation_latent=mean_latent, input_is_latent=True
            )
            print(sample.shape)
            global_i +=1
            utils.save_image(
                sample,
                f"generate/{str(global_i).zfill(6)}.png",
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--path", type=str, default='/home/huteng/zhuhaokun/4-i/pixel2style2pixel/tmp/latent_results', help="path to latent"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/huteng/zhuhaokun/4-i/stylegan2-pytorch/checkpoint/002000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    dataset = MyDataset(args.path)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            drop_last=False)

    generate(args, g_ema, device, mean_latent, dataloader)
