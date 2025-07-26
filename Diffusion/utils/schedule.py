import numpy as np

def get_beta_schedule(schedule_type, beta_start, beta_end, num_diffusion_timesteps):
    """
    Returns a beta schedule in numpy array.
    Supported schedules: linear, cosine, quadratic, sigmoid, const, jsd
    """
    if schedule_type == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_type == "cosine":
        timesteps = np.arange(num_diffusion_timesteps + 1, dtype=np.float64) / num_diffusion_timesteps
        alphas_cumprod = np.cos((timesteps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
    elif schedule_type == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif schedule_type == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = 1 / (1 + np.exp(-betas))
        betas = betas * (beta_end - beta_start) + beta_start
    elif schedule_type == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif schedule_type == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_type}")

    assert betas.shape == (num_diffusion_timesteps,)
    return betas
