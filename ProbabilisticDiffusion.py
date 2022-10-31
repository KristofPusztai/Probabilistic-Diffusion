import torch
import torch.nn as nn
import math
from utils import get_beta_schedule
from torch.nn.modules.loss import _Loss as tLoss
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        self.alphas = 1-torch.tensor(betas, dtype=torch.float)
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
                    model_outputs = self.model(inputs.type(dtype=torch.float), t)
                    # Loss Calculations
                    loss = self.loss_fn(model_outputs, z)
                    if plot_loss:
                        losses.append(loss.detach().numpy())
                    # Backwards Step
                    loss.backward()
                    tepoch.set_postfix(loss=loss.item())
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

    def sample(self, n: int, plot_intervals=None, **kwargs):
        """
        :param n:
        :type n:
        :param plot_intervals:
        :type plot_intervals:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        x_t = torch.randn((n, self.data.shape[1]))
        x = [x_t.detach().numpy()]
        for t in range(self.T-1, -1, -1):
            z = torch.randn_like(x_t)
            a_t = self.alphas[t].reshape((-1, 1))
            a_bar_t = self.alpha_bar[t].reshape((-1, 1))
            sigma_t = torch.sqrt(1-a_t)
            mean = (1 / torch.sqrt(a_t)) *\
                   (x_t - ((1 - a_t) / torch.sqrt(1 - a_bar_t) * self.model(x_t, torch.tensor([t]))))
            x_t = mean + sigma_t * z
            x.append(x_t.detach().numpy())
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
