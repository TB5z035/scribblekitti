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
import wandb
import yaml
from dataloader.semantickitti import SemanticKITTI
from network.cylinder3d import Cylinder3DProject
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearnex import patch_sklearn
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.barlow_twins_loss import BarlowTwinsLoss, MECTwinsLoss

patch_sklearn()
from sklearn.manifold import TSNE

PRETRAIN_LOSS = {
    'barlow_twins': BarlowTwinsLoss,
    'mec': MECTwinsLoss
}

class LightningTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.network = Cylinder3DProject(nclasses=self.nclasses, **config['model'])

        loss_type = self.config['pretrain_loss']['type']
        self.loss = PRETRAIN_LOSS[loss_type](self.network.feature_size, **self.config['pretrain_loss'])

        self.save_hyperparameters('config')

    def forward(self, model, fea, pos):
        _, features = model(fea, pos, len(fea))
        return features

    def training_step(self, batch, batch_idx):
        (rpz_a, fea_a, label_), (rpz_b, fea_b, label_b) = batch
        output_a = self(self.network, fea_a, rpz_a)
        output_b = self(self.network, fea_b, rpz_b)
        loss = self.loss(output_a, output_b)

        self.log('pretrain_loss', loss, prog_bar=True)
        if self.global_rank == 0:
            self.logger.experiment.log({"pretrain_loss": loss.item()})

        return loss

    def training_epoch_end(self, outputs) -> None:
        os.makedirs(os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'], self.config['logger']['name'], 'model'), exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'], self.config['logger']['name'], 'model', f'{self.current_epoch}.ckpt'))

    def validation_step(self, batch, batch_idx):
        if self.global_rank == 0:
            rpz, fea, _ = batch
            output = self(self.network, fea, rpz)
            return output.cpu()[::100]
        else:
            return None

    def validation_epoch_end(self, outputs) -> None:
        if self.global_rank == 0:
            features = torch.cat(outputs, dim=0).cpu().numpy()
            print('Number of features: ', len(features))
            feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=3).fit_transform(features)
            dirpath = os.path.join(self.config['base_dir'], 'tsne')
            os.makedirs(dirpath, exist_ok=True)
            dirpath = os.path.join(dirpath, f'{self.current_epoch}.png')
            plt.scatter(feature_embedded[:, 0], feature_embedded[:, 1], s=1)
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
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
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
    model = LightningTrainer(config)
    Trainer(logger=wandb_logger, callbacks=model.get_model_callback(), **config['trainer']).fit(model)
