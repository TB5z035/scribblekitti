import argparse
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
# import lightning.pytorch as pl
import torch
import torch.nn as nn
import yaml
from dataloader.semantickitti import SemanticKITTI
from network.cylinder3d import Cylinder3D
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
# from lightning.pytorch import Trainer
# from lightning.pytorch.loggers import WandbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from utils.consistency_loss import PartialConsistencyLoss
from utils.evaluation import compute_iou
from utils.lovasz import lovasz_softmax
from utils.LESS_loss import LESS_Loss
from collections import OrderedDict


class LightningTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.student = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.teacher = Cylinder3D(nclasses=self.nclasses, **config['model'])
        # self.student = torch.nn.DataParallel(self.student, device_ids=[0, 1]).cuda()
        # self.teacher = torch.nn.DataParallel(self.student, device_ids=[0, 1]).cuda()
        if 'load_checkpoint' in self.config:
            ckpt_path = self.config['load_checkpoint']
            state_dict = torch.load(ckpt_path)
            state_dict_stu = OrderedDict()
            state_dict_teacher = OrderedDict()
            for k,v in state_dict['state_dict'].items():
                if k.split('.')[0] == 'student':
                    state_dict_stu[k.split('student.')[1]] = v
                else:
                    state_dict_teacher[k.split('teacher.')[1]] = v
            state_dict['state_dict'] = state_dict_stu
            self.student.load_state_dict(state_dict_stu)
            state_dict['state_dict'] = state_dict_teacher
            self.teacher.load_state_dict(state_dict_teacher)
            print('loaded checkpoint from ' + ckpt_path)
        self.initialize_teacher()

        self.loss_ls = lovasz_softmax
        self.loss_cl = PartialConsistencyLoss(H=nn.CrossEntropyLoss, ignore_index=0)
        self.less_loss = LESS_Loss(H=nn.CrossEntropyLoss,alpha=self.config['Weighted_focal_loss']['alpha'],gamma=self.config['Weighted_focal_loss']['gamma'],ignore_index=0,ignore_propogated_index=self.config['Propogated_Ignore_Index'])
        self.teacher_cm = ConfusionMatrix(self.nclasses)
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
        self.update_teacher()
        student_rpz, student_fea, student_label = batch['student']
        teacher_rpz, teacher_fea, _ = batch['teacher']
        label_group = batch['label_group']      # label_group是[batch融合点数，1]的tensor，有四种值,0,1,2,3,对应如下
                                                # noting, scribbles, propogated, weak
                                                #   0         1          2         3 
        LESS_labels = batch['LESS_labels']
        batch_size = len(student_rpz)
        student_label = torch.cat(student_label, dim=0)     # label是[batch融合点数,1]的tensor

        student_output = self(self.student, student_fea, student_rpz, batch_size)       # 输出[batch融合点数,20]的tensor
        teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size)
        cl_loss = self.loss_cl(student_output, teacher_output, student_label)
        ls_loss = self.loss_ls(student_output.softmax(1), student_label, ignore=0)
        
        # loss_weak,loss_propogated = self.less_loss(student_output.softmax(1),student_label,LESS_labels,label_group)
        loss_weak,loss_propogated,loss_ls_propoageted = self.less_loss(student_output,student_label,LESS_labels,label_group)
        # loss =  4.0/6.5 * cl_loss + 4.0 / 6.5 * ls_loss + 4.0*0.5/6.5 * loss_weak + 4.0 * 2.0/6.5 * loss_propogated
        # loss =  cl_loss + ls_loss + loss_propogated + 0.25 * loss_weak
        # loss = loss.sum()
        loss = cl_loss + ls_loss + loss_propogated + 0.5 * loss_weak + loss_ls_propoageted
        # loss = cl_loss + 2 * ls_loss + 2 * loss_propogated 

        # sch = self.lr_schedulers()
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            # sch.step()
        self.log('cl_loss', cl_loss, on_epoch=True, prog_bar=True)
        self.log('ls_loss', ls_loss, on_epoch=True, prog_bar=True)
        self.log('ls_propoageted',loss_ls_propoageted,on_epoch=True, prog_bar=True)
        self.log('weak_loss',loss_weak,on_epoch=True, prog_bar=True)
        self.log('propogated_loss',loss_propogated,on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        # print('cl_loss',cl_loss,'ls_loss',ls_loss,'weak_loss',loss_weak,flush=True)
        # return {'loss': loss}
        return loss

    def validation_step(self, batch, batch_idx):
        student_rpz, student_fea, student_label = batch['student']
        teacher_rpz, teacher_fea, teacher_label = batch['teacher']
        batch_size = len(student_rpz)

        student_label = torch.cat(student_label, dim=0)
        teacher_label = torch.cat(teacher_label, dim=0)

        student_output = self(self.student, student_fea, student_rpz, batch_size)
        teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size)

        loss = self.loss_cl(student_output, teacher_output, student_label) + \
               self.loss_ls(student_output.softmax(1), student_label, ignore=0)
        # loss = loss.sum()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        mask = (teacher_label!=0).squeeze()
        self.student_cm.update(student_output.argmax(1)[mask], student_label[mask])
        self.teacher_cm.update(teacher_output.argmax(1)[mask], teacher_label[mask])

    def validation_epoch_end(self, outputs):
        student_iou, student_miou = compute_iou(self.student_cm.compute(), ignore_zero=True)
        self.student_cm.reset()
        for class_name, class_iou in zip(self.unique_name, student_iou):
            self.log('val_student_iou_{}'.format(class_name), class_iou * 100)
        self.log('val_student_miou', student_miou, on_epoch=True, prog_bar=True)

        teacher_iou, teacher_miou = compute_iou(self.teacher_cm.compute(), ignore_zero=True)
        self.teacher_cm.reset()
        for class_name, class_iou in zip(self.unique_name, teacher_iou):
            self.log('val_teacher_iou_{}'.format(class_name), class_iou * 100)
        self.log('val_teacher_miou', teacher_miou, on_epoch=True, prog_bar=True)

        if teacher_miou > student_miou:
            if teacher_miou > self.best_miou:
                self.best_miou = teacher_miou
                self.best_iou = np.nan_to_num(teacher_iou) * 100
        else:
            if student_miou > self.best_miou:
                self.best_miou = student_miou
                self.best_iou = np.nan_to_num(student_miou) * 100
        self.log('val_best_miou', self.best_miou, on_epoch=True, prog_bar=True)
        self.log('val_best_iou', self.best_iou, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), **self.config['optimizer'])
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # return [optimizer],[scheduler]
        return [optimizer]

    def setup(self, stage):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['val_dataset'])
        # here! 

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=self.train_dataset._collate_fn, **self.config['train_dataloader'])

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, collate_fn=self.val_dataset._collate_fn, **self.config['val_dataloader'])
    # here!

    def initialize_teacher(self) -> None:
        self.alpha = 0.99 # TODO: Move to config
        for p in self.teacher.parameters(): p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(alpha).add_(1 - alpha, sp.data)

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
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}-{val_teacher_miou:.2f}',
                                                  monitor='val_teacher_miou', mode='max', save_top_k=3)
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}-{val_best_miou:.2f}',
                                                  monitor='val_best_miou', mode='max', save_top_k=1)
        return [checkpoint]
    # def training_step_end(self, outputs):
 
    #     if outputs is None:
    #         return None
    #     if outputs['loss'] is None:
    #         return None
    #     self.log('train_loss', outputs['loss'].mean(), on_epoch=True, prog_bar=True)
    #     return {'epoch': self.current_epoch,
    #             'loss': outputs['loss'].mean()}
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/train/cylinder3d/cylinder3d_mt_LESS.yaml')
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

    wandb_logger = WandbLogger(config=config,
                               save_dir=config['trainer']['default_root_dir'],
                               **config['logger'])
    model = LightningTrainer(config)
    Trainer(logger=wandb_logger,
            callbacks=model.get_model_callback(),
            **config['trainer']).fit(model)
