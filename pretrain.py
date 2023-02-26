import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from dataloader.semantickitti import SemanticKITTI
from network.cylinder3d import Cylinder3D
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.manifold import TSNE
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from utils.barlow_twins_loss import BarlowTwinsLoss
from utils.consistency_loss import PartialConsistencyLoss
from utils.evaluation import compute_iou
from utils.lovasz import lovasz_softmax

import wandb


class LightningTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.network = Cylinder3D(nclasses=self.nclasses, **config['model'])

        self.loss = BarlowTwinsLoss()

        self.save_hyperparameters('config')

    def forward(self, model, fea, pos):
        _, features = model([fea.squeeze(0)], [pos.squeeze(0)], 1)
        return features

    def training_step(self, batch, batch_idx):
        (rpz_a, fea_a, label_), (rpz_b, fea_b, label_b) = batch
        output_a = self(self.network, fea_a, rpz_a)
        output_b = self(self.network, fea_b, rpz_b)
        loss = self.loss(output_a, output_b)

        self.log('pretrain_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch % 10 == 9:
            torch.save(self.network.state_dict(), 'output/scribblekitti/network.ckpt')

    def validation_step(self, batch, batch_idx):
        rpz, fea, _ = batch
        output = self(self.network, fea, rpz)
        return output

    def validation_epoch_end(self, outputs):
        features = torch.cat(outputs, dim=0).cpu().numpy()
        print(len(features))
        feature_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(features[::50])
        wandb.log({"tsne": wandb.plot.scatter(wandb.Table(data=feature_embedded, columns=['x', 'y']), 'x', 'y')})

    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(), **self.config['optimizer'])
        return [optimizer]

    def setup(self, stage):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['val_dataset'])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.config['train_dataloader'])

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.config['val_dataloader'])

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = len(dataset_config['labels'])
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)
        for i in range(self.nclasses):
            self.color_map[i, :] = torch.tensor(dataset_config['color_map'][i][::-1], dtype=torch.float32)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'])
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath,
                                                  filename='{epoch}')
        return [checkpoint]


if __name__ == '__main__':
    wandb.init("scribble-pretrain")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    config['val_dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])
    model = LightningTrainer(config)
    Trainer(logger=wandb_logger, callbacks=model.get_model_callback(), **config['trainer']).fit(model)
