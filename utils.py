import torch


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
