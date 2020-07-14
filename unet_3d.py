import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.block1_conv1 = nn.Conv3d(1, 12, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block1_conv2 = nn.Conv3d(12, 12, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block1_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.block2_conv1 = nn.Conv3d(12, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block2_conv2 = nn.Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block2_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.block3_conv1 = nn.Conv3d(60, 300, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block3_conv2 = nn.Conv3d(300, 300, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block3_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.block4_conv1 = nn.Conv3d(300, 1500, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block4_conv2 = nn.Conv3d(1500, 1500, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block4_convt = nn.ConvTranspose3d(1500, 300, kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.block5_conv0 = nn.Conv3d(600, 300, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.block5_conv1 = nn.Conv3d(300, 300, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block5_conv2 = nn.Conv3d(300, 300, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block5_convt = nn.ConvTranspose3d(300, 60, kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.block6_conv0 = nn.Conv3d(120, 60, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.block6_conv1 = nn.Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block6_conv2 = nn.Conv3d(60, 60, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block6_convt = nn.ConvTranspose3d(60, 12, kernel_size=(1, 3, 3), stride=(1, 3, 3))

        self.block7_conv0 = nn.Conv3d(24, 12, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.block7_conv1 = nn.Conv3d(12, 12, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.block7_conv2 = nn.Conv3d(12, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1))


    def center_crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_z = (layer_depth - target_size[0]) // 2
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        return layer[:, :, diff_z : (diff_z + target_size[0]), diff_y : (diff_y + target_size[1]), diff_x : (diff_x + target_size[2])]

    def forward(self, x):
        x =  F.relu(self.block1_conv1(x))
        x1 = F.relu(self.block1_conv2(x))
        x =  self.block1_pool(x1)

        x =  F.relu(self.block2_conv1(x))
        x2 = F.relu(self.block2_conv2(x))
        x =  self.block2_pool(x2)

        x =  F.relu(self.block3_conv1(x))
        x3 = F.relu(self.block3_conv2(x))
        x =  self.block3_pool(x3)

        x =  F.relu(self.block4_conv1(x))
        x =  F.relu(self.block4_conv2(x))
        x =  self.block4_convt(x)

        x3 = self.center_crop(x3, x.shape[2:])
        x =  torch.cat([x3, x], dim=1)
        x =  F.relu(self.block5_conv0(x))
        x =  F.relu(self.block5_conv1(x))
        x =  F.relu(self.block5_conv2(x))
        x =  self.block5_convt(x)

        x2 = self.center_crop(x2, x.shape[2:])
        x =  torch.cat([x2, x], dim=1)
        x =  F.relu(self.block6_conv0(x))
        x =  F.relu(self.block6_conv1(x))
        x =  F.relu(self.block6_conv2(x))
        x =  self.block6_convt(x)

        x1 = self.center_crop(x1, x.shape[2:])
        x =  torch.cat([x1, x], dim=1)
        x =  F.relu(self.block7_conv0(x))
        x =  F.relu(self.block7_conv1(x))
        x =  torch.sigmoid(self.block7_conv2(x))

        return x

