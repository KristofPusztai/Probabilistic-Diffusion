import numpy as np
import re
import torch


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def preserve_zeros(num, desired_length):
    str_num = str(round(num, desired_length))
    post_decimal = re.findall(r'\..*', str_num)[0].replace('.', '')
    while len(post_decimal) < desired_length:
        str_num += '0'
        post_decimal = re.findall(r'\..*', str_num)[0].replace('.', '')
    return str_num


def normal_grad(grid, sd):
    # Normal grad, 2 dimensions
    assert grid.shape[1] == 2, '2D gradients supported only'
    x1 = grid[:, 0]
    x2 = grid[:, 1]

    fx1 = 1 / (sd[0] * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x1 ** 2 / (2 * sd[0] ** 2))
    fx2 = 1 / (sd[1] * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x2 ** 2 / (2 * sd[1] ** 2))

    grad = torch.zeros((len(grid), 2))

    grad1 = fx2 * 1 / (sd[0] * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x1 ** 2 / (2 * sd[0] ** 2)) * (-x1 / (sd[0] ** 2))
    grad2 = fx1 * 1 / (sd[1] * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-x2 ** 2 / (2 * sd[1] ** 2)) * (-x2 / (sd[1] ** 2))

    grad[:, 0] = grad1
    grad[:, 1] = grad2
    return grad

def tweedies(data, probabilities, gradients, sd, clearance = 0.000000001):
    # Tweedies formula
    tweedies1 = sd[:, 0] * (gradients[:, 0] / (probabilities + 0.000001))
    tweedies2 = sd[:, 1] * (gradients[:, 1] / (probabilities + 0.000001))
    tweedies = data + torch.stack((tweedies1, tweedies2), dim=1)
    return tweedies

@torch.no_grad()
def estimate_distribution(diffusion, n, grid, sd, plot_idx = [0,1]):
    d = grid.shape[1]
    assert len(plot_idx) == 2, '2D plotting only'
    assert d == 2, '2D plotting only'
    means = diffusion.sample(n, no_noise=True, keep = 'last')
    probs = torch.zeros((len(grid),))
    for i in range(len(grid)):
        p1 = torch.exp(Normal(torch.tensor(0), sd[i, 0]).log_prob(grid[i, 0] - means[:, 0]))
        p2 = torch.exp(Normal(torch.tensor(0), sd[i,0]).log_prob(grid[i, 1] - means[:, 1]))
        probs[i] = torch.mean(p1 * p2, dim=0)
    return probs

@torch.no_grad()
def estimate_gradient(diffusion, n, grid, sd, plot_idx = [0,1]):
    d = grid.shape[1]
    assert len(plot_idx) == 2, '2D plotting only'
    assert d == 2, '2D gradients supported only'
    means = diffusion.sample(n, no_noise=True, keep = 'last')
    gradients = torch.zeros_like(grid)
    for i in range(len(grid)):
        gradients[i] = torch.mean(normal_grad(grid[i] - means, sd = sd[i]), dim=0)
    return gradients

