import torch
import torch.nn as nn
from utils import get_beta_schedule
from torch.nn.modules.loss import _Loss as tLoss
from torch.optim import Optimizer
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

class Diffusion:
    def __init__(self, data: torch.tensor, num_diffusion_timesteps: int,
                 beta_start: int, beta_end: int, schedule: str,
                 mean_model: nn.Module, mean_loss_fn: tLoss,
                 cov_model: nn.Module, cov_loss_fn: tLoss,
                 mean_optimizer: Optimizer, cov_optimizer: Optimizer):
        self.data = data
        self.mean_model = mean_model
        self.mean_loss_fn = mean_loss_fn
        self.cov_model = cov_model
        self.cov_loss_fn = cov_loss_fn
        self.mean_optimizer = mean_optimizer
        self.cov_optimizer = cov_optimizer
        self.num_diffusion_timesteps = num_diffusion_timesteps
        betas = get_beta_schedule(schedule, beta_start=beta_start, beta_end=beta_end,
                                  num_diffusion_timesteps=num_diffusion_timesteps)
        self.alphas = 1-torch.tensor(betas)
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def train(self, batch_size: int, epochs: int):
        for epoch in range(epochs):
            self.mean_optimizer.zero_grad()
            self.cov_optimizer.zero_grad()
            x0_ind = torch.randint(low=0, high=(len(self.data) - 1), size=(batch_size, 1))
            x0 = self.data[x0_ind]
            t = torch.randint(low=0, high=self.num_diffusion_timesteps, size=(batch_size,))
            alpha_t_bars = self.alpha_bar[t-1].reshape((-1, 1))
            z = torch.rand((batch_size, 1))
            inputs = torch.sqrt(alpha_t_bars)*x0 + torch.sqrt(1 - alpha_t_bars) * z

            inputs = torch.cat([t, inputs], dim=1)
            mean_outputs = self.mean_model(inputs)
            cov_outputs = self.cov_model(inputs)
            mean_loss = self.mean_loss_fn(mean_outputs)
            cov_loss = self.cov_loss_fn(cov_outputs)
            mean_loss.backward()
            cov_loss.backward()
            self.mean_optimizer.step()
            self.cov_optimizer.step()
        # TODO
        pass
    def forward(self, t, plot=True, s=5):
        d = self.data.shape[1]
        if plot:
            assert d == 2, 'Data is not 2d, cannot plot'
        alpha_bar = self.alpha_bar[t]
        cov = (1-alpha_bar) * torch.eye(len(self.data))
        samples = []
        for i in range(d):
            d_data = self.data[:, i]
            mu_t = d_data*torch.sqrt(alpha_bar)
            samples.append(MultivariateNormal(mu_t, cov).sample())
        data_t = torch.stack(samples, dim=1)
        if plot:
            plt.scatter(data_t[:, 0], data_t[:, 1], s=s)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Samples At Time {t}')
        return data_t