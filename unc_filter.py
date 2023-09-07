import argparse
import os
import pathlib
import sys

import h5py
import numpy as np
import torch
import yaml
from dataloader.semantickitti import SemanticKITTI
from tqdm import tqdm
from vis import vis_label

from sufield.fit import mixture_filter, BetaDistribution, GammaDistribution
from sufield.spec_cluster import geodesic_correlation_matrix, angular_correlation_matrix, spectral_cluster
from sufield.utils import plydata_to_array, construct_plydata

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/unc_filter.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    parser.add_argument('--stat_dir', default='inference')
    parser.add_argument('--save_dir', default='pseudo_label')
    parser.add_argument('--pseudo_name', default='unc')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    ds = SemanticKITTI(split='train', config=config['dataset'])
    num_classes = len(config['dataset']['labels'])
    hf = h5py.File(os.path.join(args.stat_dir, 'training_results.h5'), 'r')
    cuda = config["cuda"]
    # for all data in the train split, filter out the pseudo label
    print('Filtering the pseudo label')
    for i in tqdm(range(len(ds))):
        label_path = ds.label_paths[i] # e.g. dataset/sequences/00/scribbles/000000.label
        predicted_labels = hf[os.path.join(label_path, 'label')][()]
        uncertainty_scores = hf[os.path.join(label_path, 'unc')][()]
        lidar = ds.get_lidar(i)
        
        init_dist_a = GammaDistribution(2, 3)
        init_dist_b = GammaDistribution(3, 2)
        filter_mask = mixture_filter(uncertainty_scores, init_dist_a, init_dist_b)
        # Save pseudo-labels
        new_labels = predicted_labels[filter_mask[:, 0]]
        # true_label = learning_map_inv[scribbles].astype(np.uint32)
        path_split = label_path.split('/')
        path_split[-2] = args.pseudo_name
        unc_path = pathlib.Path(args.save_dir, *path_split[-4:])
        print(unc_path)
        # crb_path = pathlib.Path(label_path.replace('scribbles', 'unc'))
        unc_path.parents[0].mkdir(parents=True, exist_ok=True)
        new_labels.tofile(unc_path)
    hf.close()
    
# def filter_uncertainty_from_obj(dataset_root: str, stat_root: str, save_root: str, cuda=True):
#     import torch
#     for plyfile in os.listdir(dataset_root):
#         assert plyfile.endswith('.ply')
#         plyfile_path = os.path.join(dataset_root, plyfile)
#         plydata = PlyData.read(plyfile_path)
#         (vertices, rgbs, _), faces = plydata_to_array(plydata)
#         stat_path = os.path.join(stat_root, plyfile)
#         predicted_labels = np.asarray(torch.load(stat_path[:-4] + '_predicted.obj'))
#         uncertainty_scores = np.asarray(torch.load(stat_path[:-4] + '_unc.obj'))

#         init_dist_a = GammaDistribution(2, 3)
#         init_dist_b = GammaDistribution(3, 2)
#         filter_mask = mixture_filter(uncertainty_scores, init_dist_a, init_dist_b, cuda=cuda)

#         save_path = os.path.join(save_root, plyfile)
#         new_labels = predicted_labels[filter_mask]
#         construct_plydata(vertices, rgbs, new_labels).write(save_path)