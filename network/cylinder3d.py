import multiprocessing

import numba as nb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from network.modules.cylinder3d import (ReconBlock, ResBlock, ResContextBlock,
                                        UpBlock)

import spconv


class FeatureGenerator(nn.Module):
    def __init__(self,
                 in_feat=9,
                 out_feat=16, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.pretrain = False

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_feat),

            nn.Linear(in_feat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
        )

        self.compress = nn.Sequential(
            nn.Linear(256, out_feat),
            nn.ReLU(out_feat)
        )

    def forward(self, feat, coord, reverse_indices=None, shuffle=None):
        # Concatenate data
        coords = []
        for b in range(len(coord)):
            coords.append(F.pad(coord[b], (1, 0), 'constant', value=b))
        feats = torch.cat(feat, dim=0)
        coords = torch.cat(coords, dim=0)

        # Shuffle data
        if shuffle is None:
            shuffle = torch.randperm(feats.shape[0], device=feat[0].device)
        feats = feats[shuffle, :]
        if reverse_indices is not None:
            reverse_indices = reverse_indices[shuffle]
            unique_inv = reverse_indices
            coords = coords[shuffle, :]
            unique_coords = torch_scatter.scatter_max(coords, reverse_indices, dim=0)[0]
        else:
        # Unique coordinates
            if self.pretrain:
                pass
            else:
                coords = coords[shuffle, :]
                unique_coords, unique_inv = torch.unique(coords, return_inverse=True, dim=0)

        # Generate features
        feats = self.net(feats)
        if not self.pretrain:
            feats = torch_scatter.scatter_max(feats, unique_inv, dim=0)[0]
        feats = self.compress(feats)
        if self.pretrain:
            return feats, coords
        return feats, unique_coords.type(torch.int64)

class AsymmetricUNet(nn.Module):
    def __init__(self,
                 spatial_shape,
                 nclasses=20,
                 in_feat=16,
                 hid_feat=32):
        super().__init__()
        self.spatial_shape = np.array(spatial_shape)

        self.contextBlock = ResContextBlock(in_feat, hid_feat, indice_key="pre")
        self.resBlock0 = ResBlock(hid_feat, 2 * hid_feat, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock1 = ResBlock(2 * hid_feat, 4 * hid_feat, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock2 = ResBlock(4 * hid_feat, 8 * hid_feat, 0.2, pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock3 = ResBlock(8 * hid_feat, 16 * hid_feat, 0.2, pooling=True, height_pooling=False, indice_key="down5")

        self.upBlock0 = UpBlock(16 * hid_feat, 16 * hid_feat, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * hid_feat, 8 * hid_feat, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * hid_feat, 4 * hid_feat, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * hid_feat, 2 * hid_feat, indice_key="up3", up_key="down2")

        self.reconBlock = ReconBlock(2 * hid_feat, 2 * hid_feat, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * hid_feat, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        self.dropout = nn.Dropout()

    def forward(self, voxel_features, coors, batch_size):
        # print(voxel_features.shape, "voxel")
        # print(coors.shape, "coordinates")
        # print(batch_size, "batch_size")
        ret = spconv.SparseConvTensor(voxel_features, coors.int(), self.spatial_shape, batch_size)
        ret = self.contextBlock(ret)
        
        down1c, down1b = self.resBlock0(ret)
        down2c, down2b = self.resBlock1(down1c)
        down3c, down3b = self.resBlock2(down2c)
        down4c, down4b = self.resBlock3(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.reconBlock(up1e)
        up0e.features = torch.cat((up0e.features, up1e.features), 1)
        
        up0e.features = self.dropout(up0e.features)
        # print(up0e.features.shape)
        # print("up0e end")
        logits = self.logits(up0e)
        y = logits.dense()
        # print("y shape", y.shape)
        return y, up0e


class Cylinder3D(nn.Module):
    def __init__(self,
                 spatial_shape=[480,360,32],
                 nclasses=20,
                 in_feat=9,
                 hid_feat=32, downsample=True):
        super().__init__()
        self.fcnn = FeatureGenerator(in_feat=in_feat, out_feat=hid_feat//2, downsample=downsample)
        self.unet = AsymmetricUNet(spatial_shape=spatial_shape,
                                   nclasses=nclasses,
                                   in_feat=hid_feat//2,
                                   hid_feat=hid_feat)

    def forward(self, feat, coord, batch_size, unique_invs=None, shuffle=None):
        # import IPython
        # IPython.embed()
        feat, coord = self.fcnn(feat, coord, unique_invs, shuffle)
        return self.unet(feat, coord, batch_size)

class Cylinder3DProject(Cylinder3D):
    def __init__(self, spatial_shape=[480, 360, 32], nclasses=20, in_feat=9, hid_feat=32, downsample=True):
        super().__init__(spatial_shape, nclasses, in_feat, hid_feat, downsample)
        self.feature_size = 4 * hid_feat
        self.projector = spconv.SubMConv3d(4 * hid_feat, 4 * hid_feat, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
    def forward(self, feat, coord, batch_size, unique_invs=None, shuffle=None):
        y, hidden = super().forward(feat, coord, batch_size, unique_invs, shuffle)
        # import IPython
        # IPython.embed()
        
        # y.shape.shape is [1, 20, 480, 360, 32]
        # hidden.shape is [1, 128, 480, 360, 32] as sparse Tensor
        
        # feature nums may differ
        return y, self.projector(hidden).features # later.shape is [46952, 128]
        # return y, self.projector(hidden).dense().reshape(-1, 1)