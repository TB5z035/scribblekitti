# created by cyh
# 用于可视化 lasermix 的两个被混合起来的部分
import argparse
import math
import os
import pathlib
import sys

import h5py
import numpy as np
import torch
import yaml
from dataloader.semantickitti import SemanticKITTI
from tqdm import tqdm
from vis import vis_seg

from sufield.fit import mixture_filter, BetaDistribution, GammaDistribution
from sufield.spec_cluster import geodesic_correlation_matrix, angular_correlation_matrix, spectral_cluster
from sufield.utils import plydata_to_array, construct_plydata

def mix_mask_phi(rpz_1, rpz_2, bound=(0, math.pi / 2), bincount=50):
    step = (bound[1] - bound[0]) / bincount
    odd_mask_1 = torch.div(torch.arctan(torch.div(rpz_1[:, 2], rpz_1[:, 0])), step, rounding_mode='floor') % 2 == 1
    odd_mask_2 = torch.div(torch.arctan(torch.div(rpz_2[:, 2], rpz_2[:, 0])), step, rounding_mode='floor') % 2 == 1
    return odd_mask_1, odd_mask_2


def mix_mask_x(rpz_1, rpz_2, bound=(0, 50), bincount=10):
    step = (bound[1] - bound[0]) // bincount
    odd_mask_1 = torch.div(rpz_1[:, 0], step, rounding_mode='floor') % 2 == 1
    odd_mask_2 = torch.div(rpz_2[:, 0], step, rounding_mode='floor') % 2 == 1
    return odd_mask_1, odd_mask_2

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/unc_filter.yaml')
    parser.add_argument('--dataset_config_path', default='config/dataset/semantickitti.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    ds = SemanticKITTI(split='train', config=config['dataset'])
    num_classes = len(config['dataset']['labels'])
    
    # for i in tqdm(range(len(ds))):
    lidar_a = ds.get_lidar(0)
    lidar_b = ds.get_lidar(1)
    (rpz_a, label_a) = ds[0]
    (rpz_b, label_b) = ds[1]
    # 在 dim = 1 上加一列全0的列
    lidar_a = torch.cat([torch.Tensor(lidar_a), torch.zeros(lidar_a.shape[0], 1)], dim=1)
    lidar_b = torch.cat([torch.Tensor(lidar_b), torch.ones(lidar_b.shape[0], 1)], dim=1)
    print(lidar_a.shape, lidar_b.shape)
    odd_mask_a, odd_mask_b = mix_mask_phi(rpz_a, rpz_b)
    lidar = torch.cat([torch.Tensor(lidar_a[torch.logical_not(odd_mask_a)]), torch.Tensor(lidar_b[odd_mask_b])], dim=0)
    label = torch.cat([label_a[torch.logical_not(odd_mask_a)], label_b[odd_mask_b]], dim=0)
    vis_seg(lidar)