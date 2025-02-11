import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from dataloader.semantickitti import SemanticKITTI
from network.cylinder3d import Cylinder3D
from utils.consistency_loss import PartialConsistencyLoss
from utils.evaluation import compute_iou
from utils.lovasz import lovasz_softmax


class LightningTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.student = Cylinder3D(nclasses=self.nclasses, **config['model'])
        if 'load_checkpoint' in self.config['trainer']:
            ckpt_path = self.config['trainer']['load_checkpoint']
            state_dict = torch.load(ckpt_path)
            self.student.load_state_dict(state_dict)
            print('loaded checkpoint from ' + ckpt_path)

        self.loss_ls = lovasz_softmax

        self.student_cm = ConfusionMatrix(self.nclasses)
        self.best_miou = 0
        self.best_iou = np.zeros((self.nclasses-1,))

        self.save_hyperparameters('config')

    def forward(self, model, fea, pos, batch_size):
        output_voxel, _ = model(fea, pos, batch_size)
        outputs = []
        for i in range(batch_size):
            outputs.append(output_voxel[i, :, pos[i][:, 0], pos[i][:, 1], pos[i][:, 2]])
        return torch.cat(outputs, dim=1).T # (\sigma Bi*Ni, C)

    def training_step(self, batch, batch_idx):
        student_rpz, student_fea, student_label = batch
        batch_size = len(student_rpz)
        student_label = torch.cat(student_label, dim=0)

        student_output = self(self.student, student_fea, student_rpz, batch_size)
        loss = self.loss_ls(student_output.softmax(1), student_label, ignore=0)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        student_rpz, student_fea, student_label = batch
        batch_size = len(student_rpz)

        student_label = torch.cat(student_label, dim=0)

        student_output = self(self.student, student_fea, student_rpz, batch_size)

        loss = self.loss_ls(student_output.softmax(1), student_label, ignore=0)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        mask = (student_label!=0).squeeze()
        self.student_cm.update(student_output.argmax(1)[mask], student_label[mask])

    def validation_epoch_end(self, outputs):
        student_iou, student_miou = compute_iou(self.student_cm.compute(), ignore_zero=True)
        self.student_cm.reset()
        for class_name, class_iou in zip(self.unique_name, student_iou):
            self.log('val_student_iou_{}'.format(class_name), class_iou * 100)
        self.log('val_student_miou', student_miou, on_epoch=True, prog_bar=True)

        if student_miou > self.best_miou:
            self.best_miou = student_miou
            self.best_iou = np.nan_to_num(student_iou) * 100
        self.log('val_best_miou', self.best_miou, on_epoch=True, prog_bar=True)
        self.log('val_best_iou', self.best_iou, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), **self.config['optimizer'])
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
            self.color_map[i,:] = torch.tensor(dataset_config['color_map'][i][::-1], dtype=torch.float32)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['base_dir'], 'ckpt')
        os.makedirs(dirpath, exist_ok=True)
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}-{val_student_miou:.2f}',
                                                  monitor='val_student_miou', mode='max', save_top_k=3)
        return [checkpoint]


if __name__=='__main__':
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
    config['base_dir'] = base_dir

    wandb_logger = WandbLogger(config=config,
                               save_dir=config['trainer']['default_root_dir'],
                               **config['logger'])
    model = LightningTrainer(config)
    Trainer(logger=wandb_logger,
            callbacks=model.get_model_callback(),
            **config['trainer']).fit(model)
