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
            self.network.load_state_dict(state_dict)
            self.network = self.network.cuda()
            print('loaded checkpoint from ' + ckpt_path)
        
        loss_type = self.config['pretrain_loss']['type']
        self.loss = PRETRAIN_LOSS[loss_type](self.network.feature_size, **self.config['pretrain_loss'])

        self.save_hyperparameters('config')
        self.feat_cache_a = torch.zeros((12000, 9))
        self.feat_cache_b = torch.zeros((12000, 9))
        self.rpz_cache_a = torch.zeros((12000, 3))
        self.rpz_cache_b = torch.zeros((12000, 3))
        self.cache_count = 0

    def forward(self, model, fea, pos, reverse_indices=None):
        _, features = model(fea, pos, len(fea), unique_invs=reverse_indices)
        return features

    def training_step(self, batch, batch_idx):
        (rpz_a, fea_a, label_a, transform), (rpz_b, fea_b, label_b, _) = batch
        
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
        # unique_feats_a = torch_scatter.scatter_max(feats_a, unique_inv_a, dim=0)[0] # 感觉这里 max pooling 不对
        unique_feats_a = torch_scatter.scatter_mean(feats_a, unique_inv_a, dim=0)
        unique_coords_b, unique_inv_b = torch.unique(coords_b, return_inverse=True, dim=0)
        # unique_feats_b = torch_scatter.scatter_max(feats_b, unique_inv_b, dim=0)[0]
        unique_feats_b = torch_scatter.scatter_mean(feats_b, unique_inv_b, dim=0)
    
        transform = torch.cat(transform, dim=0) 
        
        # transform b by the transformation that has done by a
        
        unique_coords_b_transformed = []
        for i in unique_coords_b:
            unique_coords_b_transformed.append(F.pad(transform[i[0] * 5529600 + i[1] + (i[2] + i[3] * 360) * 480], (1, 0), 'constant', value=i[0]))
        unique_coords_b_transformed = torch.stack(unique_coords_b_transformed, dim=0) # 70308 -> 35517 unique ones
        
        unique_transformed_coords_b, unique_transformed_inv_b = torch.unique(unique_coords_b_transformed, return_inverse=True, dim=0)
        # unique_transformed_feats_b = torch_scatter.scatter_max(unique_feats_b, unique_transformed_inv_b, dim=0)[0]
        unique_transformed_feats_b = torch_scatter.scatter_mean(unique_feats_b, unique_transformed_inv_b, dim=0)        
        a_cat_b, counts = torch.unique(torch.cat([unique_coords_a, unique_transformed_coords_b]), return_counts=True, dim=0)
        intersect_coords = a_cat_b[torch.where(counts.cpu().gt(1))] # 17 unique intersects
        
        mask_a = (unique_coords_a[:, None] == intersect_coords).all(-1).any(-1) # 只有 17 个在 intersect 中
        feats_a_filtered = unique_feats_a[mask_a]
        coords_a_filtered = unique_coords_a[mask_a]
        
        # select the intersects respectively
        mask_b = np.where((unique_transformed_coords_b[:, None] == coords_a_filtered).all(-1).any(-1).cpu() == True)[0]
        feats_b_filtered = unique_transformed_feats_b[mask_b]
        coords_b_filtered = unique_transformed_coords_b[mask_b][:,:3]
        
        coords_a_filtered = coords_a_filtered[:,:3]
        
        for i in range(feats_a_filtered.shape[0]):
            self.feat_cache_a[i + self.cache_count] = feats_a_filtered[i]
            self.feat_cache_b[i + self.cache_count] = feats_b_filtered[i]
            self.rpz_cache_a[i + self.cache_count] = coords_a_filtered[i]
            self.rpz_cache_b[i + self.cache_count] = coords_b_filtered[i]
        
        # cache the features and compute loss when it's big enough
        self.cache_count += feats_a_filtered.shape[0]
        
        print("# uniquified voxels = " , self.cache_count)
        
        if self.cache_count > 10000:
            output_a = self(self.network, (self.feat_cache_a[:10000].cuda(),), (self.rpz_cache_a[:10000].cuda(),))
            output_b = self(self.network, (self.feat_cache_b[:10000].cuda(),), (self.rpz_cache_b[:10000].cuda(),))
            loss = self.loss(output_a, output_b)
            self.cache_count = 0
            self.feat_cache_a = torch.zeros((12000, 9))
            self.feat_cache_b = torch.zeros((12000, 9))
            self.rpz_cache_a = torch.zeros((12000, 3))
            self.rpz_cache_b = torch.zeros((12000, 3))
            self.log('pretrain_loss', loss, prog_bar=True)
            if self.global_rank == 0:
                self.logger.experiment.log({"pretrain_loss": loss.item()})
        else:
            return

        return loss
    
        #  这不对呐，应该有 12367 个的
        # mask_b = (unique_coords_b_transformed[:, None] == intersect_coords).all(-1).any(-1) # 47808 个在 intersect 中
        
        # In [124]: torch.unique(unique_coords_b_transformed[np.where(mask_b.cpu() == True)[0]], dim=0).shape[0]
        # Out[124]: 12367
        # In [125]: intersect_coords.shape[0]
        # Out[125]: 12367
        # In [126]: torch.unique(unique_coords_a[np.where(mask_a.cpu() == True)[0]], dim=0).shape[0]
        # Out[126]: 13
        
        # mask_a = np.isin(unique_coords_a.cpu(), intersect_coords.cpu())
        # mask_b = np.isin(unique_coords_b_transformed.cpu(), intersect_coords.cpu())
        
        # 找到对应关系
        # reorder
        # masks_a = []
        # for i in range(128):
        #     masks_a.append(mask_a)
        # mask_a = torch.stack(masks_a).reshape(len(mask_a), 128)
        # masks_b = []
        # for i in range(128):
        #     masks_b.append(mask_b)
        # mask_b = torch.stack(masks_b).reshape(len(mask_b), 128)
        # feats_a_filtered = mask_a * output_a
        # feats_b_filtered = mask_b * output_b
        
        # loss = self.loss(feats_a_filtered, feats_b_filtered)
        
        # _, ind_sorted = torch.sort(unique_inv_a, stable=True)
        # cum_sum = counts_a.cumsum(0)
        # cum_sum = torch.cat((torch.tensor([0]).cuda(), cum_sum[:-1]))
        # first_indicies_a = ind_sorted[cum_sum]
        
        # _, ind_sorted = torch.sort(unique_inv_b, stable=True)
        # cum_sum = counts_b.cumsum(0)
        # cum_sum = torch.cat((torch.tensor([0]).cuda(), cum_sum[:-1]))
        # first_indicies_b = ind_sorted[cum_sum]
        
        # 找到两个 unique 数组的并集
        # indexs = np.intersect1d(first_indicies_a.cpu(), first_indicies_b.cpu())

        # coords_a = coords_a[indexs]
        # coords_b = coords_b[indexs]
        # feats_a = feats_a[indexs]
        # feats_b
        
        # len_a = len(unique_coords_a)
        # len_b = len(unique_coords_b)
        # for i in indexs:
        #     for j in range(len(coords_a)):
        #         if unique_inv_a[j] == unique_inv_a[i] and j in indexs and j != i:
        #             unique_inv_a[j] = len_a
        #             len_a += 1
        #         if unique_inv_b[j] == unique_inv_b[i] and j in indexs and j != i:
        #             unique_inv_b[j] = len_b
        #             len_b += 1
            
        
        # import IPython
        # IPython.embed()
        
        # torch.cuda.empty_cache()
        
        # skip=True 跳过 model 里对数据的处理
        # output_a = self(self.network, feats_a_uniqued, coords_a_uniqued, skip=True)
        # output_b = self(self.network, feats_b_uniqued, coords_b_uniqued, skip=True)
        
        # output_a = self(self.network, fea_a, rpz_a, reverse_indices=unique_inv_a)
        # output_b = self(self.network, fea_b, rpz_b, reverse_indices=unique_inv_b)
        
        # loss = self.loss(output_a, output_b)

    def training_epoch_end(self, outputs) -> None:
        dirpath = os.path.join(self.config['base_dir'], 'model')
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(dirpath, f'{self.current_epoch}.ckpt'))

    def validation_step(self, batch, batch_idx):
        if self.global_rank == 0:
            rpz, fea, label = batch
            output = self(self.network, fea, rpz, reverse_indices=[])
            return [output.cpu()[::100], label[0].cpu()[::100]]
        else:
            return None

    def validation_epoch_end(self, outputs) -> None:
        if self.global_rank == 0:
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

    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(), **self.config['optimizer'])
        return [optimizer]

    def setup(self, stage):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['val_dataset'])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=self.train_dataset._collate_fn, **self.config['train_dataloader'])

    def val_dataloader(self):
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
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, save_last=True, filename='epoch-{epoch:02d}', period=1)
        return [checkpoint]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
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

    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])
    model = LightningTrainer(config=config)
    Trainer(logger=wandb_logger, callbacks=model.get_model_callback(), **config['trainer']).fit(model)
