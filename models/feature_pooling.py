import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_initializer


class ConvMLP(nn.Module):
    def __init__(self, in_features=1, out_features=1, pool_size=(7, 7)):
        super(ConvMLP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.fc1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.avg_pool(x)
        # flatten image input
        x = x.view(x.size(0), -1)
        # add hidden layer, with relu activation function
        x = F.elu(self.fc1(x))

        return x


class FeaturePooling(nn.Module):
    def __init__(self, options):
        super(FeaturePooling, self).__init__()
        self.vertices_per_mesh = options.vpm
        self.features_per_mlp = options.fpm
        self.avg_pool_size = (options.aps, options.aps)
        self.in_feats_conv1MLP = options.nf * options.aps * options.aps
        self.in_feats_conv2MLP = 2 * self.in_feats_conv1MLP
        self.in_feats_conv3MLP = 4 * self.in_feats_conv1MLP
        self.in_feats_conv4MLP = 8 * self.in_feats_conv1MLP

        self.output_features = self.vertices_per_mesh * self.features_per_mlp

        self.conv1MLP = ConvMLP(in_features=self.in_feats_conv1MLP, out_features=self.output_features,
                                pool_size=self.avg_pool_size)
        self.conv1MLP.apply(weights_initializer)

        self.conv2MLP = ConvMLP(in_features=self.in_feats_conv2MLP, out_features=self.output_features,
                                pool_size=self.avg_pool_size)
        self.conv2MLP.apply(weights_initializer)

        self.conv3MLP = ConvMLP(in_features=self.in_feats_conv3MLP, out_features=self.output_features,
                                pool_size=self.avg_pool_size)
        self.conv3MLP.apply(weights_initializer)

        self.conv4MLP = ConvMLP(in_features=self.in_feats_conv4MLP, out_features=self.output_features,
                                pool_size=self.avg_pool_size)
        self.conv4MLP.apply(weights_initializer)

    def forward(self, conv1, conv2, conv3, conv4, point_set):
        conv1_mlp_feats = self.conv1MLP(conv1)
        conv1_mlp_feats = conv1_mlp_feats.reshape(conv1_mlp_feats.size(0) * self.vertices_per_mesh,
                                                  self.features_per_mlp)
        conv2_mlp_feats = self.conv2MLP(conv2)
        conv2_mlp_feats = conv2_mlp_feats.reshape(conv2_mlp_feats.size(0) * self.vertices_per_mesh,
                                                  self.features_per_mlp)
        conv3_mlp_feats = self.conv3MLP(conv3)
        conv3_mlp_feats = conv3_mlp_feats.reshape(conv3_mlp_feats.size(0) * self.vertices_per_mesh,
                                                  self.features_per_mlp)
        conv4_mlp_feats = self.conv4MLP(conv4)
        conv4_mlp_feats = conv4_mlp_feats.reshape(conv4_mlp_feats.size(0) * self.vertices_per_mesh,
                                                  self.features_per_mlp)

        # finally aggregate with the point-set [N, features + vertex coords]
        tot_feats = torch.cat([point_set, conv1_mlp_feats, conv2_mlp_feats, conv3_mlp_feats, conv4_mlp_feats], dim=-1)

        return tot_feats
