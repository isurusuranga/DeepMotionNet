import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GraphResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin1 = nn.Linear(in_features=in_channels, out_features=out_channels // 2)
        self.conv1 = GATConv(in_channels=out_channels // 2, out_channels=out_channels // 2, heads=1, concat=False)
        self.conv2 = GATConv(in_channels=out_channels // 2, out_channels=out_channels // 2, heads=1, concat=False)
        self.lin2 = nn.Linear(in_features=out_channels // 2, out_features=out_channels)
        self.skip_conv = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm3 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.lin1.weight, gain=1)
        nn.init.constant_(self.lin1.bias, 0)
        nn.init.xavier_uniform_(self.lin2.weight, gain=1)
        nn.init.constant_(self.lin2.bias, 0)
        nn.init.xavier_uniform_(self.skip_conv.weight, gain=1)
        nn.init.constant_(self.skip_conv.bias, 0)

    def forward(self, x, edge_index):
        y = F.elu(self.pre_norm(x))
        y = self.lin1(y)
        y = F.elu(self.norm1(y))

        y = self.conv1(y, edge_index)
        y = F.elu(self.norm2(y))

        y = self.conv2(y, edge_index)
        y = F.elu(self.norm3(y))

        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return x + y


class MeshDeformationNet(nn.Module):
    def __init__(self, options):
        super(MeshDeformationNet, self).__init__()
        self.num_channels = options.nc
        # 3 for vertex position
        # 20 for total features from all conv layers
        self.fc1 = nn.Linear(3 + 20, 2 * self.num_channels)
        self.gResBlock1 = GraphResBlock(2 * self.num_channels, self.num_channels)
        self.gResBlock2 = GraphResBlock(self.num_channels, self.num_channels)
        self.gResBlock3 = GraphResBlock(self.num_channels, self.num_channels)
        self.gResBlock4 = GraphResBlock(self.num_channels, self.num_channels)
        self.gResBlock5 = GraphResBlock(self.num_channels, self.num_channels)
        self.gResBlock6 = GraphResBlock(self.num_channels, self.num_channels)
        self.gResBlock7 = GraphResBlock(self.num_channels, self.num_channels)
        self.gResBlock8 = GraphResBlock(self.num_channels, self.num_channels // 2)
        self.g_norm1 = nn.GroupNorm((self.num_channels // 2) // 8, self.num_channels // 2)

        self.fc_out = nn.Linear(self.num_channels // 2, 3)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = self.gResBlock1(x, edge_index)
        x = self.gResBlock2(x, edge_index)
        x = self.gResBlock3(x, edge_index)
        x = self.gResBlock4(x, edge_index)
        x = self.gResBlock5(x, edge_index)
        x = self.gResBlock6(x, edge_index)
        x = self.gResBlock7(x, edge_index)
        x = self.gResBlock8(x, edge_index)
        x = F.elu(self.g_norm1(x))
        x = self.fc_out(x)

        return x
