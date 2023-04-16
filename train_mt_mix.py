import argparse
import os
import shutil
import sys
from datetime import datetime

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from train_mt import LightningTrainer
from utils.timer import Timer

TIMER_SILENT = True

def mix_mask(rpz_1, rpz_2, bound=(0, 50), bincount=10):
    step = (bound[1] - bound[0]) // bincount
    odd_mask_1 = torch.div(rpz_1[:, 0], step, rounding_mode='floor') % 2 == 1
    odd_mask_2 = torch.div(rpz_2[:, 0], step, rounding_mode='floor') % 2 == 1
    return odd_mask_1, odd_mask_2


def lasermix(data_1, data_2):
    fea_batch_1, rpz_batch_1, label_batch_1 = data_1
    fea_batch_2, rpz_batch_2, label_batch_2 = data_2
    assert len(fea_batch_1) == len(fea_batch_2) == len(rpz_batch_1) == len(rpz_batch_2) == len(label_batch_1) == len(label_batch_2), f"{len(fea_batch_1)}, {len(fea_batch_2)}, {len(rpz_batch_1)}, {len(rpz_batch_2)}, {len(label_batch_1)}, {len(label_batch_2)}"
    assert len(fea_batch_1) % 2 == 0
    batch_size = len(fea_batch_1) // 2
    mixed_data = []
    for bi in range(batch_size):
        with Timer(f'lasermix {bi}', slient=TIMER_SILENT) as timer:
            fea_a, rpz_a, label_a = fea_batch_1[bi * 2], rpz_batch_1[bi * 2], label_batch_1[bi * 2]
            fea_b, rpz_b, label_b = fea_batch_2[bi * 2 + 1], rpz_batch_2[bi * 2 + 1], label_batch_2[bi * 2 + 1]
            timer.tick('split1')
            odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b)
            timer.tick('mix_mask1')
            mixed_data.append((
                torch.cat([fea_a[odd_mask_a], fea_b[torch.logical_not(odd_mask_b)]], dim=0),
                torch.cat([rpz_a[odd_mask_a], rpz_b[torch.logical_not(odd_mask_b)]], dim=0),
                torch.cat([label_a[odd_mask_a], label_b[torch.logical_not(odd_mask_b)]], dim=0),
            ))
            mixed_data.append((
                torch.cat([fea_a[torch.logical_not(odd_mask_a)], fea_b[odd_mask_b]], dim=0),
                torch.cat([rpz_a[torch.logical_not(odd_mask_a)], rpz_b[odd_mask_b]], dim=0),
                torch.cat([label_a[torch.logical_not(odd_mask_a)], label_b[odd_mask_b]], dim=0),
            ))
            timer.tick('append1')
            fea_a, rpz_a, label_a = fea_batch_1[bi * 2 + 1], rpz_batch_1[bi * 2 + 1], label_batch_1[bi * 2 + 1]
            fea_b, rpz_b, label_b = fea_batch_2[bi * 2], rpz_batch_2[bi * 2], label_batch_2[bi * 2]
            timer.tick('split2')
            odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b)
            timer.tick('mix_mask1')
            mixed_data.append((
                torch.cat([fea_a[odd_mask_a], fea_b[torch.logical_not(odd_mask_b)]], dim=0),
                torch.cat([rpz_a[odd_mask_a], rpz_b[torch.logical_not(odd_mask_b)]], dim=0),
                torch.cat([label_a[odd_mask_a], label_b[torch.logical_not(odd_mask_b)]], dim=0),
            ))
            mixed_data.append((
                torch.cat([fea_a[torch.logical_not(odd_mask_a)], fea_b[odd_mask_b]], dim=0),
                torch.cat([rpz_a[torch.logical_not(odd_mask_a)], rpz_b[odd_mask_b]], dim=0),
                torch.cat([label_a[torch.logical_not(odd_mask_a)], label_b[odd_mask_b]], dim=0),
            ))
            timer.tick('append2')

    return tuple(zip(*mixed_data))

class LightningMixTrainer(LightningTrainer):

    def forward(self, model, fea, pos, batch_size, unique_invs=None, shuffle=None):
        output_voxel, _ = model(fea, pos, batch_size, unique_invs, shuffle)
        outputs = []
        for i in range(batch_size):
            outputs.append(output_voxel[i, :, pos[i][:, 0], pos[i][:, 1], pos[i][:, 2]])
        return torch.cat(outputs, dim=1).T  # (\sigma Bi*Ni, C)

    def training_step(self, batch, batch_idx):
        with Timer('training_step', slient=TIMER_SILENT) as timer:
            self.update_teacher()
            timer.tick('update_teacher')

            student_rpz, student_fea, student_label_ori = batch['student']
            teacher_rpz, teacher_fea, _ = batch['teacher']
            batch_size = len(student_rpz)
            student_label = torch.cat(student_label_ori, dim=0)
            timer.tick('split_batch')

            unique_invs = [torch.unique(coord, return_inverse=True, dim=0)[1] for coord in student_rpz]
            count = 0
            for i in range(len(unique_invs)):
                unique_invs[i] += count
                count += len(unique_invs[i])
            unique_concat_invs = torch.cat(unique_invs)
            timer.tick('generate_unique_invs')

            shuffle = torch.randperm(unique_concat_invs.shape[0], device=unique_concat_invs[0].device)
            inv_shuffle = torch.argsort(shuffle)
            timer.tick('generate_shuffle')

            student_output = self(self.student, student_fea, student_rpz, batch_size, unique_invs=unique_concat_invs, shuffle=shuffle)
            timer.tick('student_output')
            teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size)
            timer.tick('teacher_output')
            cl_loss = self.loss_cl(student_output, teacher_output, student_label)
            ls_loss = self.loss_ls(student_output.softmax(1), student_label, ignore=0)
            timer.tick('calculate_loss')


            teacher_output_normalized = teacher_output.softmax(1)
            threshold = 0.8
            teacher_output_valid_mask = (teacher_output_normalized.max(1)[0] > threshold)
            teacher_pseudo_label = teacher_output_normalized.argmax(1)
            teacher_pseudo_label[teacher_output_valid_mask] = 0
            teacher_pseudo_label = teacher_pseudo_label[inv_shuffle]
            teacher_pseudo_label = teacher_pseudo_label[unique_concat_invs]
            teacher_pseudo_label = torch.split(teacher_pseudo_label, [len(label) for label in student_label_ori], dim=0)
            timer.tick('generate_pseudo_label')

            (mix_fea, mix_rpz, mix_label) = lasermix((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_pseudo_label))
            timer.tick('lasermix')
            # del student_fea, student_rpz, student_label_ori, teacher_fea, teacher_rpz, teacher_pseudo_label
            # del teacher_output, student_output, teacher_output_normalized, teacher_output_valid_mask
            # torch.cuda.empty_cache()

            mix_label = torch.cat(mix_label, dim=0)
            mix_output = self(self.student, mix_fea, mix_rpz, 2 * batch_size)
            timer.tick('mix_output')

            mix_loss = self.loss_ls(mix_output.softmax(1), mix_label, ignore=0)

            loss = cl_loss + ls_loss + mix_loss
            # del mix_fea, mix_rpz, mix_label, mix_output
            # torch.cuda.empty_cache()

            self.log('cl_loss', cl_loss, on_epoch=True, prog_bar=True)
            self.log('ls_loss', ls_loss, on_epoch=True, prog_bar=True)
            self.log('mix_loss', mix_loss, on_epoch=True, prog_bar=True)
            self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss


@rank_zero_only
def init(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    shutil.copy2(args.config_path, os.path.join(base_dir, 'config.yaml'))
    shutil.copy2(args.dataset_config_path, os.path.join(base_dir, 'dataset_config.yaml'))
    with open(os.path.join(base_dir, 'command'), 'w') as f:
        print(sys.argv, file=f)


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

    base_dir = os.path.join(config['trainer']['default_root_dir'], config['logger']['project'], config['logger']['name'],
                            datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    init(base_dir)
    config['base_dir'] = base_dir

    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])
    tb_logger = TensorBoardLogger(base_dir, name='tb')
    model = LightningMixTrainer(config)
    Trainer(logger=[wandb_logger, tb_logger], callbacks=model.get_model_callback(), **config['trainer']).fit(model)
