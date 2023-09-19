import argparse
import math
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
import numpy as np

TIMER_SILENT = True

# def mix_mask(rpz_1, rpz_2, bound=(0, 50), bincount=10):
#     step = (bound[1] - bound[0]) // bincount
#     odd_mask_1 = torch.div(rpz_1[:, 0], step, rounding_mode='floor') % 2 == 1
#     odd_mask_2 = torch.div(rpz_2[:, 0], step, rounding_mode='floor') % 2 == 1
#     return odd_mask_1, odd_mask_2

def mix_mask_x(rpz_1, rpz_2, bound=(0, 50), bincount=10):
    if type(bincount) is list:
        bincount = np.random.choice(bincount, size=1)[0]
    step = (bound[1] - bound[0]) // bincount
    odd_mask_1 = torch.div(rpz_1[:, 0], step, rounding_mode='floor') % 2 == 1
    odd_mask_2 = torch.div(rpz_2[:, 0], step, rounding_mode='floor') % 2 == 1
    return odd_mask_1, odd_mask_2

def mix_mask_phi(rpz_1, rpz_2, bound=(0, math.pi / 2), bincount=10):
    if type(bincount) is list:
        bincount = np.random.choice(bincount, size=1)[0]
    step = (bound[1] - bound[0]) / bincount
    odd_mask_1 = torch.div(torch.arctan(torch.div(rpz_1[:, 2], rpz_1[:, 0])), step, rounding_mode='floor') % 2 == 1
    odd_mask_2 = torch.div(torch.arctan(torch.div(rpz_2[:, 2], rpz_2[:, 0])), step, rounding_mode='floor') % 2 == 1
    return odd_mask_1, odd_mask_2



def lasermix(data_1, data_2, config):
    if config["lasermix"]["seg_strategy"] == "phi":
        mix_mask = mix_mask_phi
    else:
        mix_mask = mix_mask_x
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
            odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b, bincount=config["lasermix"]["seg_bincount"])
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

def lasermix_same_scene(data_1, data_2, config):
    if config["lasermix"]["seg_strategy"] == "phi":
        mix_mask = mix_mask_phi
    else:
        mix_mask = mix_mask_x
    fea_batch_1, rpz_batch_1, label_batch_1 = data_1
    fea_batch_2, rpz_batch_2, label_batch_2 = data_2
    assert len(fea_batch_1) == len(fea_batch_2) == len(rpz_batch_1) == len(rpz_batch_2) == len(label_batch_1) == len(label_batch_2), f"{len(fea_batch_1)}, {len(fea_batch_2)}, {len(rpz_batch_1)}, {len(rpz_batch_2)}, {len(label_batch_1)}, {len(label_batch_2)}"
    # assert len(fea_batch_1) % 2 == 0
    # batch_size = len(fea_batch_1) // 2
    batch_size = len(fea_batch_1)
    mixed_data = []
    for bi in range(batch_size):
        with Timer(f'lasermix {bi}', slient=TIMER_SILENT) as timer:
            fea_a, rpz_a, label_a = fea_batch_1[bi], rpz_batch_1[bi], label_batch_1[bi]
            fea_b, rpz_b, label_b = fea_batch_2[bi], rpz_batch_2[bi], label_batch_2[bi]
            timer.tick('split1')
            odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b, bincount=config["lasermix"]["seg_bincount"])
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
            timer.tick('append')
            # fea_a, rpz_a, label_a = fea_batch_1[bi * 2 + 1], rpz_batch_1[bi * 2 + 1], label_batch_1[bi * 2 + 1]
            # fea_b, rpz_b, label_b = fea_batch_2[bi * 2], rpz_batch_2[bi * 2], label_batch_2[bi * 2]
            # timer.tick('split2')
            # odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b)
            # timer.tick('mix_mask1')
            # mixed_data.append((
            #     torch.cat([fea_a[odd_mask_a], fea_b[torch.logical_not(odd_mask_b)]], dim=0),
            #     torch.cat([rpz_a[odd_mask_a], rpz_b[torch.logical_not(odd_mask_b)]], dim=0),
            #     torch.cat([label_a[odd_mask_a], label_b[torch.logical_not(odd_mask_b)]], dim=0),
            # ))
            # mixed_data.append((
            #     torch.cat([fea_a[torch.logical_not(odd_mask_a)], fea_b[odd_mask_b]], dim=0),
            #     torch.cat([rpz_a[torch.logical_not(odd_mask_a)], rpz_b[odd_mask_b]], dim=0),
            #     torch.cat([label_a[torch.logical_not(odd_mask_a)], label_b[odd_mask_b]], dim=0),
            # ))
            # timer.tick('append2')

    return tuple(zip(*mixed_data))

def lasermix_1(data_1, data_2, config):
    if config["lasermix"]["seg_strategy"] == "phi":
        mix_mask = mix_mask_phi
    else:
        mix_mask = mix_mask_x
    fea_batch_1, rpz_batch_1, label_batch_1 = data_1
    fea_batch_2, rpz_batch_2, label_batch_2 = data_2
    assert len(fea_batch_1) == len(fea_batch_2) == len(rpz_batch_1) == len(rpz_batch_2) == len(label_batch_1) == len(label_batch_2), f"{len(fea_batch_1)}, {len(fea_batch_2)}, {len(rpz_batch_1)}, {len(rpz_batch_2)}, {len(label_batch_1)}, {len(label_batch_2)}"
    assert len(fea_batch_1) % 2 == 0
    batch_size = len(fea_batch_1) // 2
    mixed_data = []
    for bi in range(batch_size):
        with Timer(f'lasermix {bi}', slient=TIMER_SILENT) as timer:
            fea_a, rpz_a, label_a = fea_batch_1[2 * bi], rpz_batch_1[2 * bi], label_batch_1[2 * bi]
            fea_b, rpz_b, label_b = fea_batch_2[2 * bi + 1], rpz_batch_2[2 * bi + 1], label_batch_2[2 * bi + 1]
            timer.tick('split1')
            odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b, bincount=config["lasermix"]["seg_bincount"])
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

    return tuple(zip(*mixed_data))

def lasermix_2(data_1, data_2, config):
    if config["lasermix"]["seg_strategy"] == "phi":
        mix_mask = mix_mask_phi
    else:
        mix_mask = mix_mask_x
    fea_batch_1, rpz_batch_1, label_batch_1 = data_1
    fea_batch_2, rpz_batch_2, label_batch_2 = data_2
    assert len(fea_batch_1) == len(fea_batch_2) == len(rpz_batch_1) == len(rpz_batch_2) == len(label_batch_1) == len(label_batch_2), f"{len(fea_batch_1)}, {len(fea_batch_2)}, {len(rpz_batch_1)}, {len(rpz_batch_2)}, {len(label_batch_1)}, {len(label_batch_2)}"
    assert len(fea_batch_1) % 2 == 0
    batch_size = len(fea_batch_1) // 2
    mixed_data = []
    for bi in range(batch_size):
        with Timer(f'lasermix {bi}', slient=TIMER_SILENT) as timer:
            fea_a, rpz_a, label_a = fea_batch_1[bi * 2 + 1], rpz_batch_1[bi * 2 + 1], label_batch_1[bi * 2 + 1]
            fea_b, rpz_b, label_b = fea_batch_2[bi * 2], rpz_batch_2[bi * 2], label_batch_2[bi * 2]
            timer.tick('split2')
            odd_mask_a, odd_mask_b = mix_mask(rpz_a, rpz_b, bincount=config["lasermix"]["seg_bincount"])
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

def generate_determinent_perm(rpz):
    unique_invs = [torch.unique(coord, return_inverse=True, dim=0)[1] for coord in rpz]
    count = 0
    for i in range(len(unique_invs)):
        unique_invs[i] += count
        count += len(unique_invs[i])
    unique_concat_invs = torch.cat(unique_invs)

    shuffle = torch.randperm(unique_concat_invs.shape[0], device=unique_concat_invs[0].device)
    inv_shuffle = torch.argsort(shuffle)

    return unique_concat_invs, shuffle, inv_shuffle

def generate_pseudo_labels(threshold, output, shuffle_inv, unique_inv, split_count):
    output_normalized = output.softmax(1)
    output_valid_mask = (output_normalized.max(1)[0] > threshold)
    pseudo_label = output_normalized.argmax(1)
    pseudo_label[output_valid_mask] = 0
    pseudo_label = pseudo_label[shuffle_inv]
    pseudo_label = pseudo_label[unique_inv]
    pseudo_label = torch.split(pseudo_label, split_count, dim=0)
    return pseudo_label

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
            teacher_rpz, teacher_fea, teacher_label_ori = batch['teacher']
            batch_size = len(student_rpz)
            student_label = torch.cat(student_label_ori, dim=0)
            timer.tick('split_batch')

            if 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'pseudo_label':
                student_unique_concat_invs, student_shuffle, student_inv_shuffle = generate_determinent_perm(student_rpz)
                teacher_unique_concat_invs, teacher_shuffle, teacher_inv_shuffle = generate_determinent_perm(teacher_rpz)
                timer.tick('generate_determinent_perm')
                student_output = self(self.student, student_fea, student_rpz, batch_size, unique_invs=student_unique_concat_invs, shuffle=student_shuffle)
                timer.tick('student_output')
                teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size, unique_invs=teacher_unique_concat_invs, shuffle=teacher_shuffle)
                timer.tick('teacher_output')
                threshold = self.config['lasermix']['pseudo_label_threshold']
                teacher_pseudo_label = generate_pseudo_labels(threshold, teacher_output, teacher_inv_shuffle, teacher_unique_concat_invs, [len(label) for label in teacher_label_ori])
                student_pseudo_label = generate_pseudo_labels(threshold, student_output, student_inv_shuffle, student_unique_concat_invs, [len(label) for label in student_label_ori])
                timer.tick('generate_pseudo_label')
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'scribble':
                student_output = self(self.student, student_fea, student_rpz, batch_size)
                timer.tick('student_output')
                teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size)
                timer.tick('teacher_output')
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'unc_label':
                student_output = self(self.student, student_fea, student_rpz, batch_size)
                timer.tick('student_output')
                teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size)
                timer.tick('teacher_output')
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'unc_and_pseudo':
                student_unique_concat_invs, student_shuffle, student_inv_shuffle = generate_determinent_perm(student_rpz)
                teacher_unique_concat_invs, teacher_shuffle, teacher_inv_shuffle = generate_determinent_perm(teacher_rpz)
                timer.tick('generate_determinent_perm')
                student_output = self(self.student, student_fea, student_rpz, batch_size, unique_invs=student_unique_concat_invs, shuffle=student_shuffle)
                timer.tick('student_output')
                teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size, unique_invs=teacher_unique_concat_invs, shuffle=teacher_shuffle)
                timer.tick('teacher_output')
                threshold = self.config['lasermix']['pseudo_label_threshold']
                teacher_pseudo_label = generate_pseudo_labels(threshold, teacher_output, teacher_inv_shuffle, teacher_unique_concat_invs, [len(label) for label in teacher_label_ori])
                # student_pseudo_label = generate_pseudo_labels(threshold, student_output, student_inv_shuffle, student_unique_concat_invs, [len(label) for label in student_label_ori])
                timer.tick('generate_pseudo_label')
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'same_aug':
                student_unique_concat_invs, student_shuffle, student_inv_shuffle = generate_determinent_perm(student_rpz)
                teacher_unique_concat_invs, teacher_shuffle, teacher_inv_shuffle = generate_determinent_perm(teacher_rpz)
                timer.tick('generate_determinent_perm')
                student_output = self(self.student, student_fea, student_rpz, batch_size, unique_invs=student_unique_concat_invs, shuffle=student_shuffle)
                timer.tick('student_output')
                teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size, unique_invs=teacher_unique_concat_invs, shuffle=teacher_shuffle)
                timer.tick('teacher_output')
                threshold = self.config['lasermix']['pseudo_label_threshold']
                teacher_pseudo_label = generate_pseudo_labels(threshold, teacher_output, teacher_inv_shuffle, teacher_unique_concat_invs, [len(label) for label in teacher_label_ori])
                # student_pseudo_label = generate_pseudo_labels(threshold, student_output, student_inv_shuffle, student_unique_concat_invs, [len(label) for label in student_label_ori])
                timer.tick('generate_pseudo_label')
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'same_scene':
                student_unique_concat_invs, student_shuffle, student_inv_shuffle = generate_determinent_perm(student_rpz)
                teacher_unique_concat_invs, teacher_shuffle, teacher_inv_shuffle = generate_determinent_perm(teacher_rpz)
                timer.tick('generate_determinent_perm')
                student_output = self(self.student, student_fea, student_rpz, batch_size, unique_invs=student_unique_concat_invs, shuffle=student_shuffle)
                timer.tick('student_output')
                teacher_output = self(self.teacher, teacher_fea, teacher_rpz, batch_size, unique_invs=teacher_unique_concat_invs, shuffle=teacher_shuffle)
                timer.tick('teacher_output')
                threshold = self.config['lasermix']['pseudo_label_threshold']
                teacher_pseudo_label = generate_pseudo_labels(threshold, teacher_output, teacher_inv_shuffle, teacher_unique_concat_invs, [len(label) for label in teacher_label_ori])
                # student_pseudo_label = generate_pseudo_labels(threshold, student_output, student_inv_shuffle, student_unique_concat_invs, [len(label) for label in student_label_ori])
                timer.tick('generate_pseudo_label')
                
            else:
                raise NotImplementedError

            cl_loss = self.loss_cl(student_output, teacher_output, student_label)
            ls_loss = self.loss_ls(student_output.softmax(1), student_label, ignore=0)
            timer.tick('calculate_loss')
            
            # if 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'pseudo_label':
                # threshold = self.config['lasermix']['pseudo_label_threshold']
                # teacher_pseudo_label = generate_pseudo_labels(threshold, teacher_output, teacher_inv_shuffle, teacher_unique_concat_invs, [len(label) for label in teacher_label_ori])
                # student_pseudo_label = generate_pseudo_labels(threshold, student_output, student_inv_shuffle, student_unique_concat_invs, [len(label) for label in student_label_ori])
                # timer.tick('generate_pseudo_label')

            if 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'scribble':
                (mix_fea, mix_rpz, mix_label) = lasermix((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_label_ori), self.config)
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'pseudo_label':
                (mix_fea, mix_rpz, mix_label) = lasermix((student_fea, student_rpz, student_pseudo_label), (teacher_fea, teacher_rpz, teacher_pseudo_label), self.config)
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'unc_label':
                (mix_fea, mix_rpz, mix_label) = lasermix((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_label_ori), self.config)
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'unc_and_pseudo':
                (mix_fea, mix_rpz, mix_label) = lasermix((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_pseudo_label), self.config)
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'same_aug':
                (mix_fea, mix_rpz, mix_label) = lasermix((teacher_fea, teacher_rpz, teacher_label_ori), (teacher_fea, teacher_rpz, teacher_pseudo_label), self.config)
            elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'same_scene':
                (mix_fea, mix_rpz, mix_label) = lasermix_same_scene((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_pseudo_label), self.config)
            else:
                raise NotImplementedError
            timer.tick('lasermix')
            # del student_fea, student_rpz, student_label_ori, teacher_fea, teacher_rpz, teacher_pseudo_label
            # del teacher_output, student_output, teacher_output_normalized, teacher_output_valid_mask
            # torch.cuda.empty_cache()

            mix_label = torch.cat(mix_label, dim=0)
            # 注意这里的batch_size, 如果不对, 比如少乘了2，会引发 CUDA illegal memory access 错误, 让人一下子不知道从哪里开始debug
            mix_output = self(self.student, mix_fea, mix_rpz, 2 * batch_size)
            timer.tick('mix_output')

            mix_loss = self.loss_ls(mix_output.softmax(1), mix_label, ignore=0)
            
            # if 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'scribble':
            #     (mix_fea, mix_rpz, mix_label) = lasermix_2((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_label_ori), self.config)
            # elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'pseudo_label':
            #     (mix_fea, mix_rpz, mix_label) = lasermix_2((student_fea, student_rpz, student_pseudo_label), (teacher_fea, teacher_rpz, teacher_pseudo_label), self.config)
            # elif 'lasermix' in self.config and self.config['lasermix']['mix_strategy'] == 'unc_label':
            #     (mix_fea, mix_rpz, mix_label) = lasermix_2((student_fea, student_rpz, student_label_ori), (teacher_fea, teacher_rpz, teacher_label_ori), self.config)
            # else:
            #     raise NotImplementedError
            # timer.tick('lasermix')
            # # del student_fea, student_rpz, student_label_ori, teacher_fea, teacher_rpz, teacher_pseudo_label
            # # del teacher_output, student_output, teacher_output_normalized, teacher_output_valid_mask
            # # torch.cuda.empty_cache()
            # mix_label = torch.cat(mix_label, dim=0)
            # mix_output = self(self.student, mix_fea, mix_rpz, batch_size)
            # timer.tick('mix_output')

            # mix_loss += self.loss_ls(mix_output.softmax(1), mix_label, ignore=0)
            

            if self.current_epoch >= self.config['lasermix'].get('ignore', 0):
                loss = cl_loss + ls_loss + self.config['lasermix'].get('mix_loss_coe', 1) * mix_loss
            else:
                loss = cl_loss + ls_loss
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

# def record()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--name', default='no name')
    parser.add_argument('--dataset_config_path', default='config/dataset/semantickitti.yaml')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(args.dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))

    config['logger']['name'] = args.config_path.split('/')[-1][:-5]
    # config['logger']['name'] = args.name

    base_dir = os.path.join(config['trainer']['default_root_dir'], config['logger']['project'], config['logger']['name'],
                            datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    init(base_dir)
    config['base_dir'] = base_dir

    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])
    tb_logger = TensorBoardLogger(base_dir, name='tb')
    model = LightningMixTrainer(config)
    Trainer(logger=[wandb_logger, tb_logger], callbacks=model.get_model_callback(), **config['trainer']).fit(model)