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
            self.network.load_state_dict(state_dict, strict=False)
            self.network = self.network.cuda()
            print('loaded checkpoint from ' + ckpt_path)
        
        loss_type = self.config['pretrain_loss']['type']
        self.loss = PRETRAIN_LOSS[loss_type](self.network.feature_size, **self.config['pretrain_loss'])

        self.save_hyperparameters('config')

    def forward(self, model, fea, pos, reverse_indices=None):
        _, features, features2, features3 = model(fea, pos, len(fea), unique_invs=reverse_indices)
        return features, features2, features3

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
        
        # for voxel-wise contrastive loss
        # (rpz_a, fea_a, label_a, _), (rpz_b, fea_b, label_b, unique_inv) = batch
        # unique_inv = torch.cat(unique_inv, dim=0)
        # output_a = self(self.network, fea_a, rpz_a, reverse_indices=unique_inv)
        # output_b = self(self.network, fea_b, rpz_b, reverse_indices=unique_inv)
        # loss = self.loss(output_a, output_b)
        
        unique_coords_a, unique_inv_a = torch.unique(coords_a, return_inverse=True, dim=0)
        unique_feats_a = torch_scatter.scatter_mean(feats_a, unique_inv_a, dim=0)
        unique_coords_b, unique_inv_b = torch.unique(coords_b, return_inverse=True, dim=0)
        unique_feats_b = torch_scatter.scatter_mean(feats_b, unique_inv_b, dim=0)
    
        transform = torch.cat(transform, dim=0)
        
        unique_coords_b_transformed = []
        for i in unique_coords_b:
            unique_coords_b_transformed.append(F.pad(transform[i[0] * 5529600 + i[1] + (i[2] + i[3] * 360) * 480], (1, 0), 'constant', value=i[0]))
        unique_coords_b_transformed = torch.stack(unique_coords_b_transformed, dim=0)
        
        unique_transformed_coords_b, unique_transformed_inv_b = torch.unique(unique_coords_b_transformed, return_inverse=True, dim=0)
        unique_transformed_feats_b = torch_scatter.scatter_mean(unique_feats_b, unique_transformed_inv_b, dim=0)        
        a_cat_b, counts = torch.unique(torch.cat([unique_coords_a, unique_transformed_coords_b]), return_counts=True, dim=0)
        intersect_coords = a_cat_b[torch.where(counts.cpu().gt(1))] 
        
        mask_a = (unique_coords_a[:, None].cpu() == intersect_coords.cpu()).all(-1).any(-1).cuda()
        feats_a_filtered = unique_feats_a[mask_a]
        coords_a_filtered = unique_coords_a[mask_a]
        
        # select intersect points respectively
        mask_b = np.where((unique_transformed_coords_b[:, None].cpu() == coords_a_filtered.cpu()).all(-1).any(-1) == True)[0]
        feats_b_filtered = unique_transformed_feats_b[mask_b]
        coords_b_filtered = unique_transformed_coords_b[mask_b] 
        
        # import IPython
        # IPython.embed()
        
        assert feats_a_filtered.shape[0] == feats_b_filtered.shape[0], "filtered voxel number doesn't match"
        
        mask_scene0 = np.where(unique_coords_a[:,0].cpu() == 0)[0]
        fea_a_scene0 = feats_a_filtered[:mask_scene0[-1]+1]
        coo_a_scene0 = coords_a_filtered[:mask_scene0[-1]+1, 1:]
        fea_b_scene0 = feats_b_filtered[:mask_scene0[-1]+1]
        coo_b_scene0 = coords_b_filtered[:mask_scene0[-1]+1, 1:]
        
        fea_a_scene1 = feats_a_filtered[mask_scene0[-1]+1:]
        coo_a_scene1 = coords_a_filtered[mask_scene0[-1]+1:, 1:]
        fea_b_scene1 = feats_b_filtered[mask_scene0[-1]+1:]
        coo_b_scene1 = coords_b_filtered[mask_scene0[-1]+1:, 1:]
        
        output_a, out_a1, out_a2 = self(self.network, (fea_a_scene0, fea_a_scene1), (coo_a_scene0, coo_a_scene1))
        output_b, out_b1, out_b2 = self(self.network, (fea_b_scene0, fea_b_scene1), (coo_b_scene0, coo_b_scene1))
        # for deep contrastive learning
        # loss = self.loss(output_a, output_b) + self.loss(out_a1, out_b1) + self.loss(out_a2, out_b2) * 0.2
        loss = self.loss(output_a, output_b)
        self.log('# voxels', feats_a_filtered.shape[0], prog_bar=True)
        if self.global_rank == 0:
            self.logger.experiment.log({"pretrain_loss": loss.item()})
        return loss

    def training_epoch_end(self, outputs) -> None:
        dirpath = os.path.join(self.config['base_dir'], 'model')
        os.makedirs(dirpath, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(dirpath, f'{self.current_epoch}.ckpt'))

    def validation_step(self, batch, batch_idx):
        # if 'load_checkpoint' in self.config:
        # ckpt_path = self.config['load_checkpoint']
        # state_dict = torch.load(ckpt_path, map_location='cpu')
        # self.network.load_state_dict(state_dict, strict=False)
        # self.network = self.network.cuda()
        # print('loaded checkpoint from ' + ckpt_path)
        if self.global_rank == 0:
            rpz, fea, label = batch
            output, _, _ = self(self.network, fea, rpz)
            return [output.cpu(), label[0].cpu()]
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
            # dirpath = "rst_new"
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
    torch.cuda.set_device(3)
    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])
    model = LightningTrainer(config=config)
    Trainer(logger=wandb_logger, callbacks=model.get_model_callback(), **config['trainer']).fit(model)
