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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, 1024)
        x = x.view(-1, 256)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        # self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        # self.conv2 = torch.nn.Conv1d(512, 256, 1)
        # 64 + 256 = 320
        self.conv2 = torch.nn.Conv1d(320, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        # self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        # x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        x = x.view(batchsize, n_pts, 256)
        return x, trans, trans_feat


class FeatureGenerator(nn.Module):
    def __init__(self,
                 in_feat=9,
                 out_feat=16, pretrain=False, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.pretrain = pretrain

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
        
        self.net2 = PointNetDenseCls()

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
        if self.pretrain == True and reverse_indices is None:
            # feats = self.net2(feats[:, :3].reshape(1, 3, feats.shape[0]))[0]
            # feats = self.compress(feats.reshape(feats.shape[1], feats.shape[2]))
            feats = self.net(feats)
            feats = self.compress(feats)
            return feats, coords
        else:
            # Shuffle data
            if shuffle is None:
                shuffle = torch.randperm(feats.shape[0], device=feat[0].device)
            if reverse_indices is not None:
                if self.pretrain == False:
                    reverse_indices = reverse_indices[shuffle]
                    feats = feats[shuffle, :]
                    coords = coords[shuffle, :]
                unique_inv = reverse_indices
                unique_coords = torch_scatter.scatter_max(coords, reverse_indices, dim=0)[0]
            else:
                feats = feats[shuffle, :]
                coords = coords[shuffle, :]
                unique_coords, unique_inv = torch.unique(coords, return_inverse=True, dim=0) 
            # feats = []
            # max_len = 0
            # for i in range(len(feat)):
            #     if feat[i].shape[0] > max_len:
            #         max_len = feat[i].shape[0]
            # for i in range(len(feat)):
            #     feats.append(F.pad(feat[i][:, :3].reshape(1, 3, feat[i].shape[0]), (0, max_len - feat[i].shape[0]), 'constant', value=0))
            # feats = torch.cat(feats, dim=0)
            # feats = self.net2(feats)[0]
            # feats_ = []
            # for i in range(len(feat)):
            #     feats_.append(feats[i][:feat[i].shape[0]])
            # feats_ = torch.cat(feats_, dim=0)
            # feats, indexs = torch_scatter.scatter_max(feats_, unique_inv, dim=0)
            feats = self.net(feats)
            feats, _ = torch_scatter.scatter_max(feats, unique_inv, dim=0)
            feats = self.compress(feats)
            if torch.isnan(feats).any() or torch.isnan(unique_coords).any():
                import IPython
                IPython.embed()
            return feats, unique_coords.type(torch.int64)

class AsymmetricUNet(nn.Module):
    def __init__(self,
                 spatial_shape,
                 nclasses=20,
                 in_feat=16,
                 hid_feat=32,
                 pretrain=False):
        super().__init__()
        
        self.pretrain = pretrain
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
        # self.reconBlock2 = ReconBlock(4 * hid_feat, 4 * hid_feat, indice_key="recon2")
        # self.reconBlock3 = ReconBlock(8 * hid_feat, 8 * hid_feat, indice_key="recon3")

        self.logits = spconv.SubMConv3d(4 * hid_feat, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        
        self.dropout = nn.Dropout()
        # self.dropout2 = nn.Dropout()
        # self.dropout3 = nn.Dropout()

    def forward(self, voxel_features, coors, batch_size):
        # import IPython
        # IPython.embed()
        ret = spconv.SparseConvTensor(voxel_features, coors.int(), self.spatial_shape, batch_size)
        ret = self.contextBlock(ret)
        
        # N > 0 assert faild. CUDA kernel launch blocks must be positive, but got N = 0
        # 输入维度太少了就会在以下过程中出现这个问题
        down1c, down1b = self.resBlock0(ret)
        down2c, down2b = self.resBlock1(down1c)
        down3c, down3b = self.resBlock2(down2c)
        down4c, down4b = self.resBlock3(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        # up3e0 = self.reconBlock3(up3e)
        # up3e0.features = torch.cat((up3e0.features, up3e.features), 1)
        # up3e0.features = self.dropout3(up3e0.features)
        
        # add another layer of projector
        # up2e0 = self.reconBlock2(up2e)
        # up2e0.features = torch.cat((up2e0.features, up2e.features), 1)
        # up2e0.features = self.dropout2(up2e0.features)
        
        up0e = self.reconBlock(up1e)
        up0e.features = torch.cat((up0e.features, up1e.features), 1)
        up0e.features = self.dropout(up0e.features)
        
        if self.pretrain == False:
            logits = self.logits(up0e)
            y = logits.dense()
        else:
            y = None
        # return y, up0e, up2e0, up3e0
        if torch.isnan(up0e.features).any():
            import IPython
            IPython.embed()
        return y, up0e, None, None


class Cylinder3D(nn.Module):
    def __init__(self,
                 spatial_shape=[480,360,32],
                 nclasses=20,
                 in_feat=9,
                 hid_feat=32, pretrain=False, downsample=True):
        super().__init__()
        self.fcnn = FeatureGenerator(in_feat=in_feat, out_feat=hid_feat//2, pretrain=pretrain, downsample=downsample)
        self.unet = AsymmetricUNet(spatial_shape=spatial_shape,
                                   nclasses=nclasses,
                                   in_feat=hid_feat//2,
                                   hid_feat=hid_feat,
                                   pretrain=pretrain)

    def forward(self, feat, coord, batch_size, unique_invs=None, shuffle=None):
        feat, coord = self.fcnn(feat, coord, unique_invs, shuffle)
        return self.unet(feat, coord, batch_size)

class Cylinder3DProject(Cylinder3D):
    def __init__(self, spatial_shape=[480, 360, 32], nclasses=20, in_feat=9, hid_feat=32, pretrain=True, downsample=True):
        super().__init__(spatial_shape, nclasses, in_feat, hid_feat, pretrain, downsample)
        self.feature_size = 4 * hid_feat
        self.projector = spconv.SubMConv3d(4 * hid_feat, 4 * hid_feat, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)
        # self.projector2 = spconv.SubMConv3d(8 * hid_feat, 8 * hid_feat, indice_key="logit2", kernel_size=3, stride=1, padding=1,
        #                                 bias=True)
        # self.projector3 = spconv.SubMConv3d(16 * hid_feat, 16 * hid_feat, indice_key="logit3", kernel_size=3, stride=1, padding=1,
        #                                 bias=True)
    def forward(self, feat, coord, batch_size, unique_invs=None, shuffle=None):
        y, hidden, hidden2, hidden3 = super().forward(feat, coord, batch_size, unique_invs, shuffle)
        # if torch.isnan(self.projector(hidden).features).any():
        #     import IPython
        #     IPython.embed()
        # return y, self.projector(hidden).features, self.projector2(hidden2).features, self.projector3(hidden3).features
        return y, self.projector(hidden).features, None, None