import os
import yaml
import h5py
import argparse
import pathlib
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from train import LightningTrainer
from collections import OrderedDict

class LightningEvaluator(LightningTrainer):
    def __init__(self, config):
        super().__init__(config)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def test_dataloader(self):
        return self.val_dataloader()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/test.yaml')
    parser.add_argument('--dataset_config_path', default='config/dataset/semantickitti.yaml')
    parser.add_argument('--ckpt_path', default='/home/yujc/scribble/scribblekitti/output/scribblekitti_LESS/cylinder3d_mt_LESS/20230807-10:55:30/ckpt/epoch=13-val_teacher_miou=57.57.ckpt')
    args = parser.parse_args()

    config =  yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    config['val_dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])



    state_dict = torch.load(args.ckpt_path)
    state_dict_stu = OrderedDict()
    state_dict_teacher = OrderedDict()
    for k,v in state_dict['state_dict'].items():
        if k.split('.')[0] == 'student':
            state_dict_stu[k] = v
        else:
            state_dict_teacher[k] = v
    state_dict['state_dict'] = state_dict_stu
    torch.save(state_dict,'./best_stu.pt')
    # self.teacher.load_state_dict(state_dict_teacher)

    trainer = Trainer(logger=wandb_logger, **config['trainer'])
    model = LightningEvaluator.load_from_checkpoint('./best_stu.pt', config=config)
    trainer.test(model)