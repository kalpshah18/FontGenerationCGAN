import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import pandas as pd
from scipy import linalg


class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=98):
        super(Generator, self).__init__()
        self.input_dim = latent_dim + condition_dim
        self.fc = nn.Linear(self.input_dim, 7 * 7 * 128)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        return self.conv_blocks(x)


def load_generator(weights_path, device, latent_dim=100, condition_dim=98):
    gen = Generator(latent_dim, condition_dim).to(device)
    state = torch.load(weights_path, map_location=device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


def generate_font_images(generator, latent_dim, device, style_flags=None, num_chars=94, num_samples=94):
    """
    Generate `num_samples` images by sampling random latent vectors and random character conditions.
    """
    images = []
    with torch.no_grad():
        for i in range(num_samples):
            # random latent per sample
            latent = torch.randn(1, latent_dim, device=device)

            # random character index
            idx = int(torch.randint(0, num_chars, (1,)).item())

            # style flags: either provided fixed vector or random binary flags per sample
            if style_flags is None:
                style_tensor = (torch.randint(0, 2, (4,)).float()).to(device)
            else:
                style_tensor = torch.tensor(style_flags, dtype=torch.float32, device=device)

            one_hot = torch.zeros(num_chars, device=device)
            one_hot[idx] = 1.0
            condition = torch.cat((one_hot, style_tensor)).unsqueeze(0)

            gen_img = generator(latent, condition)
            img = gen_img.squeeze().cpu().numpy()
            img = (img + 1.0) / 2.0  # to [0,1]
            img = (img * 255.0).astype(np.uint8)
            img = np.stack([img, img, img], axis=2)  # to RGB
            images.append(Image.fromarray(img))
    return images


def load_real_images_from_csv(csv_path, num_samples=94):
    df = pd.read_csv(csv_path)
    # find pixel columns (they are named '1'..'784')
    pixel_cols = [c for c in df.columns if c.isdigit() or (c.isnumeric())]
    if len(pixel_cols) == 0:
        # try the usual pattern
        pixel_cols = [str(i) for i in range(1, 785)]

    # sample random rows
    sampled = df.sample(n=num_samples, replace=False, random_state=42)
    images = []
    for _, row in sampled.iterrows():
        pixels = row[pixel_cols].values.astype(np.float32)
        img = pixels.reshape(28, 28)
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img = (img * 255.0).astype(np.uint8)
        img = np.stack([img, img, img], axis=2)
        images.append(Image.fromarray(img))
    return images


def get_inception_activations(images, device, batch_size=16):
    # Use the weights enum for newer torchvision versions
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
    model.eval()

    # register hook on avgpool to capture features
    feats = []

    def _hook(module, inp, out):
        feats.append(out.detach().cpu())

    handle = model.avgpool.register_forward_hook(_hook)

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_t = torch.stack([preprocess(im) for im in batch], dim=0).to(device)
        feats.clear()
        with torch.no_grad():
            _ = model(batch_t)
        # feats[0] shape: (N, 2048, 1, 1)
        f = feats[0].squeeze(-1).squeeze(-1)
        all_feats.append(f.cpu().numpy())

    handle.remove()
    activations = np.concatenate(all_feats, axis=0)
    return activations


def calculate_fid(act1, act2):
    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    sigma2 = np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(np.real(fid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="cgan_generator_weights.pth")
    parser.add_argument("--real_csv", required=True, help="Path to 94_character_TMNIST.csv")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=94, help="Number of generated/real samples to use for FID")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if not os.path.exists(args.generator):
        raise SystemExit(f"Generator weights not found: {args.generator}")
    if not os.path.exists(args.real_csv):
        raise SystemExit(f"Real CSV not found: {args.real_csv}")

    gen = load_generator(args.generator, device, latent_dim=args.latent_dim)
    gen_images = generate_font_images(gen, args.latent_dim, device, num_chars=94, num_samples=args.num_samples)
    real_images = load_real_images_from_csv(args.real_csv, num_samples=len(gen_images))

    print("Extracting Inception activations for generated images...")
    act_gen = get_inception_activations(gen_images, device)

    print("Extracting Inception activations for real images...")
    act_real = get_inception_activations(real_images, device)

    fid_value = calculate_fid(act_real, act_gen)
    print(f"FID score (real vs generated): {fid_value:.4f}")


if __name__ == "__main__":
    main()
