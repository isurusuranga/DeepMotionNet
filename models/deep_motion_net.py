import torch.nn as nn
from utils import weights_initializer
from .cnn_img_encoder import CustomEncoder
from .feature_pooling import FeaturePooling
from .mesh_deformation_net import MeshDeformationNet


class DeepMotionNet(nn.Module):
    def __init__(self, options):
        super(DeepMotionNet, self).__init__()
        self.imgEncoder = CustomEncoder(options)
        self.imgEncoder.apply(weights_initializer)

        self.featurePooling = FeaturePooling(options)
        self.featurePooling.apply(weights_initializer)

        self.meshDeformationNet = MeshDeformationNet(options)

    def forward(self, data):
        x, edge_index, img, gantry_angle = data.x, data.edge_index, data.img, data.gantry_angle

        conv1, conv2, conv3, conv4 = self.imgEncoder(img, gantry_angle)
        x = self.featurePooling(conv1, conv2, conv3, conv4, x)
        x = self.meshDeformationNet(x, edge_index)

        return x
