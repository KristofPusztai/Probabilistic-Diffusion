import torch
import torch.nn as nn
import math
import numpy as np
from utils import get_beta_schedule
from torch.nn.modules.loss import _Loss as tLoss
from torch.optim import Optimizer
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt


class Diffusion:
    def __init__(self, data: torch.tensor, num_diffusion_timesteps: int,
                 beta_start: int, beta_end: int, schedule: str,
                 model: nn.Module, loss_fn: tLoss,
                 optimizer: Optimizer):
        self.data = data
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.T = num_diffusion_timesteps
        betas = get_beta_schedule(schedule, beta_start=beta_start, beta_end=beta_end,
                                  num_diffusion_timesteps=num_diffusion_timesteps)
        self.alphas = 1-torch.tensor(betas)
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def train(self, batch_size: int, epochs: int, plot_loss: bool = True):
        # TODO: Test
        # TODO: Add in TQDM progress bar and loss outputs
        n = len(self.data)
        d = self.data.shape[1]
        batch_in_epoch = math.ceil(n/batch_size)
        if plot_loss:
            losses = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            possible_indx = torch.tensor(range(0,n), dtype=torch.float)
            for batch in range(batch_in_epoch):
                # Batch Sample
                sample_size = min(batch_size, len(possible_indx))
                x0_ind = torch.multinomial(possible_indx, sample_size)
                # Updating Possible Index Choices After Sampling Without Replacement
                possible_indx = torch.tensor([i for i in possible_indx if i not in x0_ind])
                x0 = self.data[x0_ind]
                t = torch.randint(low=0, high=self.T, size=(sample_size,)).reshape(-1, 1)
                alpha_t_bars = self.alpha_bar[t-1].reshape((-1, 1))
                z = torch.rand((batch_size, d))
                # Set Up Inputs
                inputs = torch.sqrt(alpha_t_bars)*x0 + torch.sqrt(1 - alpha_t_bars) * z
                model_outputs = self.model(inputs.type(dtype=torch.float), t)
                # Loss Calculations
                loss = self.loss_fn(model_outputs, z)
                if plot_loss:
                    losses.append(loss.detach().numpy())
                # Backwards Step
                loss.backward()
                self.optimizer.step()
        if plot_loss:
            x_ax = range(0, len(losses))
            plt.plot(x_ax, losses)
            plt.xlabel('Batch Iteration')
            plt.ylabel('Batch Loss')
            plt.show()
            pass

    def forward(self, t, plot=True, **kwargs):
        """
        :param t:
        :type t:
        :param plot:
        :type plot:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        d = self.data.shape[1]
        if plot:
            assert d == 2, 'Data is not 2d, cannot plot'
        alpha_bar = self.alpha_bar[t]
        cov = (1-alpha_bar)
        samples = []
        for i in range(d):
            d_data = self.data[:, i]
            mu_t = d_data*torch.sqrt(alpha_bar)
            samples.append(torch.normal(mu_t, cov))
        data_t = torch.stack(samples, dim=1)
        if plot:
            plt.scatter(data_t[:, 0], data_t[:, 1], **kwargs)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Samples At Time {t}')
        return data_t

    def sample(self, n, plot_intervals=None, sigma_mixture=0, **kwargs):
        # TODO: Test
        d = self.data.shape[1]
        x_t = MultivariateNormal(torch.tensor(np.zeros(d), dtype=torch.float), torch.eye(d)).rsample(torch.Size([n]))
        x = x_t.detach()
        for t in range(self.T-1, -1, -1):
            if t > 1:
                z = MultivariateNormal(torch.tensor(np.zeros(d), dtype=torch.float), torch.eye(d)).rsample(torch.Size([n]))
            else:
                z = torch.tensor(np.zeros((n, d)), dtype=torch.float)
            a_t = self.alphas[t]
            a_bar_t = self.alpha_bar[t]
            sigma_t = (1-a_t)*(1-self.alpha_bar[t-1]*sigma_mixture - a_bar_t*(1-sigma_mixture))/(1-a_bar_t)
            x_t = (1 / torch.sqrt(a_t)) * \
                  (x_t - (1-a_t)/torch.sqrt(1 - a_bar_t)) * \
                self.model(x_t, torch.tensor(np.ones((len(x_t), 1))*t, dtype=torch.float)) + sigma_t * z
            x = torch.cat([x_t.detach(), x])
            if plot_intervals:
                assert plot_intervals > 0, 'Plot Intervals Must Be Greater Than 0'
                assert x_t.shape[1] == 2, 'Data is not 2d, cannot plot'
                if (t % plot_intervals) == 0:
                    plt.scatter(x_t[:, 0].detach().numpy(), x_t[:, 1].detach().numpy(), **kwargs)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title(f'Samples At Time {t}')
                    plt.show()
        return x
