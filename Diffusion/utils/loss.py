import torch

def noise_estimation_loss(model, x0, t, noise, betas, keepdim=False):
    """
    L_simple = || epsilon - epsilon_theta(x_t, t) ||^2
    """
    alpha = (1.0 - betas).cumprod(dim=0)
    at = alpha.index_select(0, t).view(-1, 1, 1, 1)
    xt = at.sqrt() * x0 + (1.0 - at).sqrt() * noise

    pred_noise = model(xt, t.float())

    if keepdim:
        return (pred_noise - noise).square().sum(dim=(1, 2, 3))
    else:
        return (pred_noise - noise).square().sum(dim=(1, 2, 3)).mean()


loss_registry = {
    'simple': noise_estimation_loss,
}
