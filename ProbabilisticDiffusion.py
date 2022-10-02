import numpy as np
import torch
import torch.nn as nn


class Diffusion():
    def __init__(self, data: torch.tensor , betas:np.array,  model: nn.Module):
        self.data = data
        self.model = model
        self.alphas = 1-betas
        self.aplha_bar = np.cumprod(self.alphas)

    def train(self, batch_size, betas):
        # TODO
        pass