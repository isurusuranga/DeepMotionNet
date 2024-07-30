import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomEncoder(nn.Module):
    def __init__(self, options):
        super(CustomEncoder, self).__init__()

        self.nf = options.nf

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(2, self.nf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(self.nf, 2*self.nf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(2*self.nf, 4*self.nf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(4*self.nf, 8*self.nf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv1_norm = nn.GroupNorm(self.nf // 8, self.nf)
        self.conv2_norm = nn.GroupNorm((2*self.nf) // 8, 2*self.nf)
        self.conv3_norm = nn.GroupNorm((4*self.nf) // 8, 4*self.nf)
        self.conv4_norm = nn.GroupNorm((8*self.nf) // 8, 8*self.nf)

    def forward(self, x, gantry_angle):
        gantry_angle = gantry_angle.reshape(gantry_angle.size(0), 1, 1, 1)
        # repeat => [batch_size, 1, 256, 256]
        g_rep_input = gantry_angle.repeat(1, 1, x.shape[2], x.shape[3])
        # concat on channel dim to generate tensor [16, 2, 256, 256]
        updated_input_tensor = torch.cat((x, g_rep_input), 1)

        conv1 = F.elu(self.conv1_norm(self.conv1(updated_input_tensor)))
        conv2 = F.elu(self.conv2_norm(self.conv2(self.pool(conv1))))
        conv3 = F.elu(self.conv3_norm(self.conv3(self.pool(conv2))))
        conv4 = F.elu(self.conv4_norm(self.conv4(self.pool(conv3))))

        return conv1, conv2, conv3, conv4
