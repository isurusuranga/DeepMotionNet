import torch.nn as nn
from .discrete_laplacian import DiscreteLaplacianLoss


class CustomMultiTaskLoss(nn.Module):
    def __init__(self, options):
        super(CustomMultiTaskLoss, self).__init__()
        self.weightBalance = options.wbl
        self.laplacian = DiscreteLaplacianLoss()
        self.l1Loss = nn.L1Loss()

    def forward(self, predictedCoords, groundtruthCoords, inputCoords, edge_index):
        laplacianLoss = self.laplacian(predictedCoords, inputCoords, edge_index)
        lossL1 = self.l1Loss(predictedCoords, groundtruthCoords)

        return (lossL1 + (self.weightBalance * laplacianLoss)), lossL1, laplacianLoss
