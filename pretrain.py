import argparse
import os
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearnex import patch_sklearn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pointnet2_ops._ext as _ext

import wandb
from dataloader.semantickitti import SemanticKITTI, Baseline
from network.cylinder3d import Cylinder3DProject, Cylinder3D
from utils.barlow_twins_loss import BarlowTwinsLoss, MECTwinsLoss, VICReg, EMPLoss

patch_sklearn()
from sklearn.manifold import TSNE
import torch_scatter

PRETRAIN_LOSS = {
    'barlow_twins': BarlowTwinsLoss,
    'mec': MECTwinsLoss,
    'vicreg': VICReg,
    'EMP': EMPLoss
}

import matplotlib

# rgb = [[0, 0, 0], [152, 223, 138], [255, 187, 120], [140, 86, 75], [214, 39, 40], [148, 103, 189], [23, 190, 207], [247, 182, 210], [219, 219, 141], [202, 185, 52],
# [200, 54, 131], [78, 71, 183], [255, 127, 14], [153, 98, 156], [158, 218, 229], [178, 127, 135], [146, 111, 194], [112, 128, 144], [227, 119, 194], [94, 106, 211]]
rgb = [[0, 0, 0], [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120], [188, 189, 34], [140, 86, 75], [255, 152, 150],
       [214, 39, 40], [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207], [178, 76, 76], [247, 182, 210], [66, 188, 102],
       [219, 219, 141], [140, 57, 197], [202, 185, 52], [51, 176, 203]]
rgb = np.array(rgb)/255.
cmp = matplotlib.colors.ListedColormap(rgb,"")


class LightningTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.momentum = 0.996
        self.network = Cylinder3DProject(nclasses=self.nclasses, downsample=False, **config['model'])
        if 'load_checkpoint' in self.config:
            ckpt_path = self.config['load_checkpoint']
            state_dict = torch.load(ckpt_path, map_location='cpu')
            self.network.load_state_dict(state_dict, strict=False)
            self.network = self.network.cuda()
            print('loaded checkpoint from ' + ckpt_path)
        
        loss_type = self.config['pretrain_loss']['type']
        self.loss = PRETRAIN_LOSS[loss_type](self.network.feature_size, **self.config['pretrain_loss'])

        # self.save_hyperparameters('config')

    def forward(self, model, fea, pos, reverse_indices=None):
        _, features, features2, features3 = model(fea, pos, len(fea), unique_invs=reverse_indices)
        return features, features2, features3


    def train_cluster_step_remove_ground(self, batch, batch_idx):
        (rpz_a, fea_a), (rpz_b, fea_b), clusters = batch
        
        # one thing is to try shelter the clusters and only keep those with more than \epsilon points
        # and also, build clusters again to make them contain more points
        
        rpz_a_filtered = []
        rpz_b_filtered = []
        fea_a_filtered = []
        fea_b_filtered = []
        clusters_filtered = []
        accum_cluster = -1
        for i in range(len(clusters)):
            mask_ground = np.where(clusters[i] > 0)
            cluster_remove_ground = clusters[i][mask_ground]
            fea_a_remove_ground = fea_a[i][mask_ground]
            fea_b_remove_ground = fea_b[i][mask_ground]
            rpz_a_remove_ground = rpz_a[i][mask_ground]
            rpz_b_remove_ground = rpz_b[i][mask_ground]
            uni, inv, cnts = np.unique(cluster_remove_ground, return_inverse=True, return_counts=True)
            mask_fil = np.where(cnts[inv] >= 2)
            fea_a_filtered.append(fea_a_remove_ground[mask_fil])
            fea_b_filtered.append(fea_b_remove_ground[mask_fil])
            rpz_a_filtered.append(rpz_a_remove_ground[mask_fil])
            rpz_b_filtered.append(rpz_b_remove_ground[mask_fil])
            inv_filtered = inv[mask_fil]
            _, inv_f = np.unique(inv_filtered, return_inverse=True)
            inv_f += (accum_cluster + 1)
            accum_cluster = np.max(inv_f)
            clusters_filtered.append(torch.from_numpy(inv_f).long())
            
        cluster_inv = torch.cat(clusters_filtered, axis=0)
        
        output_a, _, _ = self(self.network, fea_a_filtered, rpz_a_filtered, reverse_indices=cluster_inv.cuda())
        output_b, _, _ = self(self.network, fea_b_filtered, rpz_b_filtered, reverse_indices=cluster_inv.cuda())
        if torch.isnan(output_a).any() or torch.isnan(output_b).any():
            import IPython
            IPython.embed()
        loss = self.loss(output_a, output_b)
        if torch.isnan(loss).any():
            import IPython
            IPython.embed()
        self.logger.experiment.log({"#clusters": torch.max(cluster_inv)})
        self.logger.experiment.log({"pretrain_loss": loss.item()})
        return {"loss": loss}
        
            
        
        # remove ground
        fea_a_filtered = (fea_a[0][np.where(clusters[0] > 0)], fea_a[1][np.where(clusters[1] > 0)])
        rpz_a_filtered = (rpz_a[0][np.where(clusters[0] > 0)], rpz_a[1][np.where(clusters[1] > 0)])
        fea_b_filtered = (fea_b[0][np.where(clusters[0] > 0)], fea_b[1][np.where(clusters[1] > 0)])
        rpz_b_filtered = (rpz_b[0][np.where(clusters[0] > 0)], rpz_b[1][np.where(clusters[1] > 0)])
        clusters[0][:] -= 1
        clusters[1][np.where(clusters[1] > 0)] += np.max(clusters[0])
        
        cluster_inv = torch.from_numpy(np.concatenate((clusters[0][np.where(clusters[0] >= 0)], clusters[1][np.where(clusters[1] >= 0)]), axis=0)).long()
        
        output_a, _, _ = self(self.network, fea_a_filtered, rpz_a_filtered, reverse_indices=cluster_inv.cuda())
        output_b, _, _ = self(self.network, fea_b_filtered, rpz_b_filtered, reverse_indices=cluster_inv.cuda())
        if torch.isnan(output_a).any() or torch.isnan(output_b).any():
            import IPython
            IPython.embed()
        loss = self.loss(output_a, output_b)
        if torch.isnan(loss).any():
            import IPython
            IPython.embed()
        self.log('#clusters', torch.max(cluster_inv), prog_bar=True)
        self.logger.experiment.log({"pretrain_loss": loss.item()})
        return loss
        
    def train_cluster_step(self, batch, batch_idx):
        (rpz_a, fea_a, label_a), (rpz_b, fea_b, label_b), cluster = batch
        
        assert cluster[0] is not None, "cluster is None"
        
        # concat cluster, set all ground to 0 to remove them later
        cluster[0][np.where(cluster[0] > 0)] -= 1
        cluster[1][np.where(cluster[1] > 0)] -= 1
        cluster[0][:] -= np.min(cluster[0])
        cluster[1][:] -= np.min(cluster[1])
        cluster[1][:] += (np.max(cluster[0]) + 1)
        cluster_inv = torch.from_numpy(np.concatenate((cluster[0], cluster[1]), axis=0)).long()
        
        # make cluster as unique_inv tensor, mask ground and pass it to network
        output_a, _, _ = self(self.network, fea_a, rpz_a, reverse_indices=cluster_inv.cuda())
        output_b, _, _ = self(self.network, fea_b, rpz_b, reverse_indices=cluster_inv.cuda())
        
        if torch.isnan(output_a).any() or torch.isnan(output_b).any():
            import IPython
            IPython.embed()
        loss = self.loss(output_a, output_b)
        if torch.isnan(loss).any():
            import IPython
            IPython.embed()
        self.log('#clusters', torch.max(cluster_inv), prog_bar=True)
        self.logger.experiment.log({"pretrain_loss": loss.item()})
        return loss
        
    def train_fps_step(self, batch, batch_idx):
        (rpz_a, fea_a, xyzs), (rpz_b, fea_b, _) = batch
        fea_a_filtered = []
        rpz_a_filtered = []
        fea_b_filtered = []
        rpz_b_filtered = []
        for i in range(len(xyzs)):
            pidx =  _ext.furthest_point_sampling(xyzs[i].reshape((1, xyzs[i].shape[0], 3)).contiguous(), 4096).reshape(4096).long()
            fea_a_filtered.append(fea_a[i][pidx])
            rpz_a_filtered.append(rpz_a[i][pidx])
            fea_b_filtered.append(fea_b[i][pidx])
            rpz_b_filtered.append(rpz_b[i][pidx])
    
        output_a, _, _ = self(self.network, fea_a_filtered, rpz_a_filtered)
        output_b, _, _ = self(self.network, fea_b_filtered, rpz_b_filtered)
        
        loss = self.loss(output_a, output_b)
        self.logger.experiment.log({"pretrain_loss": loss.item()})
        if torch.isnan(loss).any():
            import IPython
            IPython.embed()
        return {"loss": loss}
    
    def train_transform_voxel_step(self, batch, batch_idx):
        spatial_shape = [120, 90, 8]
        # spatial_shape = [60, 45, 4]
        (rpz_a, fea_a, transform), (rpz_b, fea_b, _) = batch
        
        coords_a = []
        for b in range(len(rpz_a)):
            coords_a.append(F.pad(rpz_a[b], (1, 0), 'constant', value=b))
        feats_a = torch.cat(fea_a, dim=0)
        coords_a = torch.cat(coords_a, dim=0)
        
        coords_b = []
        for b in range(len(rpz_b)):
            coords_b.append(F.pad(rpz_b[b], (1, 0), 'constant', value=b))
        feats_b = torch.cat(fea_b, dim=0)
        coords_b = torch.cat(coords_b, dim=0)
        
        unique_coords_a, unique_inv_a = torch.unique(coords_a, return_inverse=True, dim=0)
        unique_coords_a_idx = (unique_coords_a[:, 0] * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] + unique_coords_a[:, 1] + (unique_coords_a[:, 2] + unique_coords_a[:, 3] * spatial_shape[1]) * spatial_shape[0]).cpu().numpy()
        # unique_feats_a = torch_scatter.scatter_mean(feats_a, unique_inv_a, dim=0)
        unique_feats_a = torch_scatter.scatter_max(feats_a, unique_inv_a, dim=0)[0]
        
        for i in range(len(rpz_a)):
            transform[i][:] += i * spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        transform = np.concatenate(transform)
        
        idx = coords_b[:, 0] * spatial_shape[0] * spatial_shape[1] * spatial_shape[2] + coords_b[:, 1] + (coords_b[:, 2] + coords_b[:, 3] * spatial_shape[1]) * spatial_shape[0]
        coords_b_idx_transformed = transform[idx.cpu().numpy()]
        
        intersect_coords = np.intersect1d(coords_b_idx_transformed, unique_coords_a_idx)
        mask_a = np.isin(unique_coords_a_idx, intersect_coords)
        feats_a_filtered = unique_feats_a[mask_a]
        coords_a_filtered = unique_coords_a[mask_a]
        coords_idx_filtered = unique_coords_a_idx[mask_a]
        
        unique_transformed_coords_b, unique_transformed_inv_b = np.unique(coords_b_idx_transformed, return_inverse=True, axis=0)
        # unique_transformed_feats_b = torch_scatter.scatter_mean(feats_b, torch.from_numpy(unique_transformed_inv_b).cuda(), dim=0)
        unique_transformed_feats_b = torch_scatter.scatter_max(feats_b, torch.from_numpy(unique_transformed_inv_b).cuda(), dim=0)[0]
        idx = []
        for i in coords_idx_filtered:
            idx.append(np.where(unique_transformed_coords_b == i)[0][0])
        mask_b = np.array(idx)
        feats_b_filtered = unique_transformed_feats_b[mask_b]
        
        assert feats_a_filtered.shape[0] == feats_b_filtered.shape[0], "filtered voxel number doesn't match"
        
        fea_a_ = []
        coo_a_ = []
        fea_b_ = []
        coo_b_ = []
        last_idx = 0
        for i in range(len(rpz_a)):
            mask_idx = np.where(coords_a_filtered[:,0].cpu() == i)[0][-1] + 1
            fea_a_.append(feats_a_filtered[last_idx:mask_idx])
            fea_b_.append(feats_b_filtered[last_idx:mask_idx])
            coo_a_.append(coords_a_filtered[last_idx:mask_idx, 1:])
        
        output_a, _, _ = self(self.network, fea_a_, coo_a_)
        output_b, _, _ = self(self.network, fea_b_, coo_a_)
        
        loss = self.loss(output_a, output_b)
        self.logger.experiment.log({"#clusters": feats_a_filtered.shape[0]})
        self.logger.experiment.log({"pretrain_loss": loss})
        return {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        # return self.train_fps_step(batch, batch_idx)
        # return self.train_transform_voxel_step(batch, batch_idx)
        return self.train_cluster_step_remove_ground(batch, batch_idx)

    def training_epoch_end(self, outputs) -> None:
        dirpath = os.path.join(self.config['base_dir'], 'model')
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(dirpath, f'{self.current_epoch}.ckpt'))

    def validation_step(self, batch, batch_idx):
        # if self.global_rank == 0:
        rpz, fea, label = batch
        output, _, _ = self(self.network, fea, rpz)
        return [output.cpu(), label[0].cpu()]
        # else:
        #     return None

    def validation_epoch_end(self, outputs) -> None:
        # if self.global_rank == 0:
        feats = [item for sublist in outputs for item in sublist][0::2]
        labels = [item for sublist in outputs for item in sublist][1::2]
        features = torch.cat(feats, dim=0).cpu().numpy()
        colors = torch.cat(labels, dim=0).cpu().numpy() + 1
        print('Number of features: ', len(features))
        feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=3).fit_transform(features)
        dirpath = os.path.join(self.config['base_dir'], 'tsne')
        os.makedirs(dirpath, exist_ok=True)
        dirpath = os.path.join(dirpath, f'{self.current_epoch}.png')
        plt.scatter(feature_embedded[:, 0], feature_embedded[:, 1], c=colors, cmap=cmp, s=1)
        plt.savefig(dirpath, dpi=600)
        plt.cla()
        dirpath = os.path.join(self.config['base_dir'], 'model')
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(dirpath, f'{self.current_epoch}.ckpt'))
        return {"val_loss": 30-self.current_epoch}

    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(), **self.config['optimizer'])
        return [optimizer]

    def setup(self, stage):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['val_dataset'])

    def train_dataloader(self):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        return DataLoader(dataset=self.train_dataset, collate_fn=self.train_dataset._collate_fn, **self.config['train_dataloader'])

    def val_dataloader(self):
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['val_dataset'])
        return DataLoader(dataset=self.val_dataset, collate_fn=self.val_dataset._collate_fn, **self.config['val_dataloader'])

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = len(dataset_config['labels'])
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)
        for i in range(self.nclasses):
            self.color_map[i, :] = torch.tensor(dataset_config['color_map'][i][::-1], dtype=torch.float32)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['base_dir'], 'ckpt')
        os.makedirs(dirpath, exist_ok=True)
        checkpoint = pl.callbacks.ModelCheckpoint(filepath=os.path.join(dirpath, 'epoch-{epoch:02d}')) # dirpath=dirpath, save_last=True, filename='epoch-{epoch:02d}', period=1
        return checkpoint
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/pretrain/pretrain_bt_cluster.yaml')
    parser.add_argument('--dataset_config_path', default='config/dataset/semantickitti.yaml')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(args.dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))

    config['logger']['name'] = args.config_path.split('/')[-1][:-5]

    base_dir = os.path.join(config['trainer']['default_root_dir'], config['logger']['project'], config['logger']['name'], datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    os.makedirs(base_dir, exist_ok=True)
    shutil.copy2(args.config_path, os.path.join(base_dir, 'config.yaml'))
    shutil.copy2(args.dataset_config_path, os.path.join(base_dir, 'dataset_config.yaml'))
    with open(os.path.join(base_dir, 'command'), 'w') as f:
        print(sys.argv, file=f) 
    config['base_dir'] = base_dir
    wandb_logger = WandbLogger(save_dir=config['trainer']['default_root_dir'], **config['logger']) # config = config
    model = LightningTrainer(config=config)
    Trainer(logger=wandb_logger, checkpoint_callback=model.get_model_callback(), **config['trainer']).fit(model) # callbacks=model.get_model_callback()
