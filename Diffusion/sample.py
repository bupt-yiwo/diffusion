import os
import time
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from model.unet import Model
from diffusion.denoising import ddpm_steps, generalized_steps
from utils.schedule import get_beta_schedule
from diffusion.ema import EMAHelper
from utils.logger import setup_logger
from utils.util import load_config, dict2namespace

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--sampler', type=str, default=None, choices=["ddpm", "ddim"])
    parser.add_argument('--eta', type=float, default=None, help="DDIM eta (0 = deterministic)")
    args = parser.parse_args()

    config_dict = load_config(args.config)
    config = dict2namespace(config_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(name="sample")

    # Resolve args from config if not specified
    ckpt_path = args.ckpt or config.sampling.ckpt
    save_dir = args.save_dir or config.sampling.save_dir
    n_samples = args.n_samples or config.sampling.n_samples
    sampler = args.sampler or config.sampling.sampler
    eta = args.eta if args.eta is not None else config.sampling.eta

    # Model
    model = Model(config).to(device)
    ema_helper = EMAHelper(mu=config.training.ema_decay)
    ema_helper.register(model)

    # Load checkpoint
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    ema_helper.load_state_dict(ckpt['ema'])
    ema_helper.ema(model)
    model.eval()

    # Beta schedule
    betas = get_beta_schedule(
        schedule_type=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
    )
    betas = torch.from_numpy(betas).float().to(device)

    # Sampling
    os.makedirs(save_dir, exist_ok=True)
    start = time.time()
    with torch.no_grad():
        z = torch.randn(n_samples, config.model.in_channels, config.data.image_size, config.data.image_size).to(device)
        seq = list(range(betas.shape[0]))
        if sampler == "ddpm":
            samples, _ = ddpm_steps(z, seq, model, betas, var_type=config.diffusion.var_type)
        elif sampler == "ddim":
            samples, _ = generalized_steps(z, seq, model, betas, eta=eta)

    for i, img in enumerate(tqdm(samples[-1], desc="Saving images")):
        save_image(img, os.path.join(save_dir, f"sample_{i:03d}.png"), normalize=True)

    elapsed = time.time() - start
    logger.info(f"Saved {n_samples} samples to {save_dir}/")
    logger.info(f"Total time: {elapsed:.2f}s, Avg per image: {elapsed / n_samples:.3f}s")

if __name__ == '__main__':
    main()
