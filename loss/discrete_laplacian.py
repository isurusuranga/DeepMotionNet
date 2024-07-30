import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class DiscreteLaplacianLoss(nn.Module):
    def __init__(self):
        super(DiscreteLaplacianLoss, self).__init__()

    def forward(self, predictedTensor, inputTensor, edge_index):
        row, col = edge_index

        delta_dash = scatter_mean((predictedTensor[row] - predictedTensor[col]), row, dim=0,
                                  dim_size=predictedTensor.size(0))
        delta = scatter_mean((inputTensor[row] - inputTensor[col]), row, dim=0, dim_size=inputTensor.size(0))

        return torch.mean(torch.pow(delta_dash - delta, 2).sum(1))
    