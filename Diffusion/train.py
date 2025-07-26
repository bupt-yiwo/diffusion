import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from model.unet import Model
from diffusion.denoising import ddpm_steps, generalized_steps
from utils.schedule import get_beta_schedule
from diffusion.ema import EMAHelper
from utils.logger import setup_logger
from datasets.dataset import ImageFolderDataset
from utils.loss import noise_estimation_loss
from utils.util import load_config, dict2namespace
from torchvision.utils import make_grid
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume_ckpt', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    config_dict = load_config(args.config)
    config = dict2namespace(config_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(logdir=config.training.log_dir, name='train')
    writer = SummaryWriter(log_dir=os.path.join(config.training.log_dir, 'tensorboard'))

    # Dataset
    dataset = ImageFolderDataset(
        root_dir=config.data.path,
        image_size=config.data.image_size,
        random_flip=config.data.random_flip
    )
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4)

    # Model
    model = Model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

    # EMA
    ema_helper = EMAHelper(mu=config.training.ema_decay)
    ema_helper.register(model)

    # Beta schedule
    betas = get_beta_schedule(
        schedule_type=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
    )
    betas = torch.from_numpy(betas).float().to(device)

    # Resume support
    step = 0
    start_epoch = 0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        logger.info(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema_helper.load_state_dict(ckpt['ema'])
        step = ckpt.get('step', 0)
        start_epoch = ckpt.get('epoch', 0)

    model.train()
    for epoch in range(start_epoch, config.training.n_epochs):
        epoch_start_time = time.time()
        with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
            for x in pbar:
                x = x.to(device)
                t = torch.randint(0, betas.shape[0], (x.size(0),), device=device).long()
                noise = torch.randn_like(x)

                loss = noise_estimation_loss(model, x, t, noise, betas)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema_helper.update(model)

                step += 1
                pbar.set_postfix(loss=loss.item(), step=step)
                writer.add_scalar("train/loss", loss.item(), global_step=step)

                if step % config.training.save_interval == 0:
                    os.makedirs(config.training.ckpt_dir, exist_ok=True)
                    torch.save({
                        'model': model.state_dict(),
                        'ema': ema_helper.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'epoch': epoch
                    }, os.path.join(config.training.ckpt_dir, f"ckpt_{step}.pt"))

                    # save sample
                    ema_helper.ema(model)
                    x_t = torch.randn_like(x[:4])
                    seq = list(range(betas.shape[0]))
                    samples, _ = generalized_steps(x_t, seq, model, betas)
                    save_dir = os.path.join(config.training.log_dir, 'samples')
                    os.makedirs(save_dir, exist_ok=True)
                    grid_path = os.path.join(save_dir, f"sample_{step}.png")
                    save_image(samples[-1], grid_path, normalize=True)
                    grid = make_grid(samples[-1], nrow=4, normalize=True)
                    writer.add_image("sample/generated_grid", grid, global_step=step)
                    model.train()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s with {len(dataloader)} batches.")

    writer.close()

if __name__ == '__main__':
    main()
