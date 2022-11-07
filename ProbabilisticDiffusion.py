import torch
import torch.nn as nn
import math
from utils import get_beta_schedule, preserve_zeros, normal_grad
from torch.nn.modules.loss import _Loss as tLoss
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union


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
        self.alphas = 1-torch.tensor(betas, dtype=self.data.dtype)
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def train(self, batch_size: int, epochs: int, plot_loss: bool = True):
        """
        :param batch_size:
        :type batch_size:
        :param epochs:
        :type epochs:
        :param plot_loss:
        :type plot_loss:
        :return:
        :rtype:
        """
        n = len(self.data)
        batch_in_epoch = math.ceil(n/batch_size)
        if plot_loss:
            losses = []
        with tqdm(range(epochs)) as tepoch:
            for epoch in tepoch:
                possible_indx = torch.tensor(range(0, n), dtype=torch.float)
                for batch in range(batch_in_epoch):
                    self.optimizer.zero_grad()
                    # Batch Sample
                    sample_size = min(batch_size, len(possible_indx))
                    x0_ind = torch.multinomial(possible_indx, sample_size)
                    # Updating Possible Index Choices After Sampling Without Replacement
                    possible_indx = torch.tensor([i for i in possible_indx if i not in x0_ind])
                    x0 = self.data[x0_ind]
                    t = torch.randint(0, self.T, size=(sample_size // 2 + 1,))
                    t = torch.cat([t, self.T - t - 1], dim=0)[:batch_size].long()
                    alpha_t_bars = self.alpha_bar[t].reshape((-1, 1))
                    z = torch.randn_like(x0)
                    # Set Up Inputs
                    inputs = torch.sqrt(alpha_t_bars) * x0 + torch.sqrt(1 - alpha_t_bars) * z
                    model_outputs = self.model(inputs, t)
                    # Loss Calculations
                    loss = self.loss_fn(model_outputs, z)
                    tqdm_loss = preserve_zeros(loss.item(), 3)
                    tepoch.set_postfix(loss=tqdm_loss)
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

    def forward(self, t: int, plot: bool = True, **kwargs):
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

    def sample(self, n: int, plot_intervals: Union[None, int]=None, no_noise: bool=False,
               keep: Union[None, str]=None, **kwargs):
        """
        :param n:
        :type n:
        :param plot_intervals:
        :type plot_intervals:
        :param no_noise:
        :type no_noise:
        :param keep: Accepts None, 'all', 'last', specifies what sampled values to keep
        :type keep: Union[None, str]
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        x_t = torch.randn((n, self.data.shape[1]))
        x = None
        if keep == 'all':
            x = [x_t.detach()]
        if no_noise:
            title = 'No Noise Samples At Time '
        else:
            title = 'Samples At Time '
        for t in range(self.T-1, -1, -1):
            z = torch.randn_like(x_t)
            a_t = self.alphas[t].reshape((-1, 1))
            a_bar_t = self.alpha_bar[t].reshape((-1, 1))
            if no_noise:
                sigma_t = 0
            else:
                sigma_t = torch.sqrt(1-a_t)
            mean = (1 / torch.sqrt(a_t)) *\
                   (x_t - ((1 - a_t) / torch.sqrt(1 - a_bar_t) * self.model(x_t, torch.tensor([t]))))
            x_t = mean + sigma_t * z
            if keep == 'all':
                detached_xt = x_t.detach()
                x.append(detached_xt)
            if plot_intervals:
                assert plot_intervals > 0, 'Plot Intervals Must Be Greater Than 0'
                assert x_t.shape[1] == 2, 'Data is not 2d, cannot plot'
                if (t % plot_intervals) == 0:
                    if keep != 'all':
                        detached_xt = x_t.detach()
                    plt.scatter(detached_xt[:, 0], detached_xt[:, 1], **kwargs)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title(title + str(t))
                    plt.show()
        if keep == 'last':
            if plot_intervals:
                x = detached_xt
            else:
                x = x_t.detach()
        return x

    @torch.no_grad()
    def estimate_distribution(self, n, grid):
        d = grid.shape[1]
        prior_points = self.sample(n, no_noise=True)[self.T - 1]
        a_t = self.alphas[1]
        a_bar_t = self.alpha_bar[1]
        means = (1 / torch.sqrt(a_t)) * \
                (prior_points - ((1 - a_t) / torch.sqrt(1 - a_bar_t) * self.model(prior_points, torch.tensor([1]))))
        probs = None
        for mean in means:
            cov = torch.eye(d) * (1 - self.alpha_bar[self.T-1])
            p = torch.exp(MultivariateNormal(mean, cov).log_prob(grid))
            if probs is not None:
                probs += p
            else:
                probs = p
        probs = probs/n
        return probs
