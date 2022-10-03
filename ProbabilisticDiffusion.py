import torch
import torch.nn as nn
from utils import get_beta_schedule


class Diffusion():
    def __init__(self, data: torch.tensor, num_diffusion_timesteps: int,
                 beta_start: int, beta_end: int,schedule: str,  model: nn.Module):
        self.data = data
        self.model = model
        self.num_diffusion_timesteps = num_diffusion_timesteps
        betas = get_beta_schedule(schedule, beta_start=beta_start, beta_end=beta_end,
                                  num_diffusion_timesteps=num_diffusion_timesteps)
        self.alphas = 1-torch.tensor(betas)
        self.aplha_bar = torch.cumprod(self.alphas)

    def train(self, batch_size, num_diffusion_timesteps, betas):
        x0_ind = torch.randint(low=0, high=(len(self.data) - 1), size=batch_size)
        x0 = self.data[x0_ind]
        t = torch.randint(low=0, high=num_diffusion_timesteps, size=(batch_size, 1))
        alpha_t_bars = self.aplha_bar[t-1].reshape((-1,1))
        z = torch.rand((batch_size, 1))
        input = torch.sqrt(alpha_t_bars)*x0 + torch.sqrt(1 - alpha_t_bars) * z
        input = torch.cat((t, input), dim=1)
        # TODO
        pass