import torch.nn.functional as F

import argparse
import os
import pathlib

import h5py
import torch
import yaml
from dataloader.semantickitti import SemanticKITTI
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from train_mt import LightningTrainer


class LightningTester(LightningTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_save_file(self, save_path):
        self.f = h5py.File(save_path, 'w')

    def test_step(self, batch, batch_idx):
        for m in self.teacher.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print(f"Warning: module {m.__class__.__name__} set to train")
                m.train()
        rpz, fea, _ = batch['teacher']
        batch_size = len(rpz)
        
        multiround_batched_scores = []
        
        for i in range(self.config["unc_round"]):
            # print(f"Info: uncertainty round #{i}")
            output_gpu = self(self.teacher, fea, rpz, batch_size)
            output = output_gpu.cpu()
            scores = F.softmax(output, dim=1)
            multiround_batched_scores.append(scores)
            
        # batch_size * point_count * class_count
        multiround_batched_scores = torch.stack(multiround_batched_scores)

        batched_labels = multiround_batched_scores.sum(dim=0).argmax(dim=1)  # Label
        batched_scores = multiround_batched_scores.var(dim=0)  # Uncertainty
        print("uncertainty sum: ", batched_scores.sum())
        # conf, pred = torch.max(output.softmax(1), dim=1)
        # conf = conf.cpu().detach().numpy()
        # pred = pred.cpu().detach().numpy()

        key = os.path.join(self.test_dataset.label_paths[batch_idx])
        unc_key, label_key = os.path.join(key, 'unc'), os.path.join(key, 'label')
        self.f.create_dataset(unc_key, data=batched_scores)
        self.f.create_dataset(label_key, data=batched_labels)

    def setup(self, stage):
        super().setup(stage)
        self.test_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        # self.test_dataset.split = 'test'
        # self.teacher.eval()
        # for m in self.teacher.modules():
        #     if m.__class__.__name__.startswith('Dropout'):
        #         print(f"Warning: module {m.__class__.__name__} set to train")
        #         m.train()


    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=self.test_dataset._collate_fn, **self.config['test_dataloader'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    parser.add_argument('--checkpoint_path', default='output/training.ckpt')
    parser.add_argument('--save_dir', default='output')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(args.dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))

    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])

    trainer = Trainer(logger=wandb_logger, **config['trainer'])
    model = LightningTester.load_from_checkpoint(args.checkpoint_path, config=config)
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model._load_save_file(os.path.join(args.save_dir, 'training_results.h5'))
    trainer.test(model)

# def unc_inference(model, dataloader, config, rank=0, world_size=1):
#     """
#     DONE
#     Run multiple inference and obtain uncertainty results for each point
#     And for each scene:
#     - save the statistics of the uncertainty in obj file (torch save)
#     - save the predicted labelds in obj file (torch save)

#     required config:
#     - unc_round
#     - unc_result_dir
#     - test_batch_size

#     call unc_render to obtain ply files
#     """

#     # Inference should be done on one GPU
#     if world_size > 1 and rank > 0:
#         return
#     device = rank

#     # Set model to eval mode except for the last dropout layer
#     model.eval()
#     for m in model.modules():
#         if m.__class__.__name__.startswith('Dropout'):
#             print(f"Warning: module {m.__class__.__name__} set to train")
#             m.train()

#     # Read scene names
#     # Scenes from training datasets
#     # with open('splits/scannet/scannetv2_train.txt') as f:
#     #     names = sorted([i.strip() for i in f.readlines()])

#     with torch.no_grad():
#         scene_cnt = 0
#         for step, batched_data in enumerate(dataloader):
#             print(f"Info: {step}/{len(dataloader)} done")

#             # Load batched data
#             batched_coords, batched_feats, _ = batched_data["teacher"]
#             # batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5
#             batch_size = len(batched_coords)

#             multiround_batched_scores = []

#             # Feed forward for multiple rounds
#             # batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))
#             for i in range(config.unc_round):
#                 print(f"Info: ---> feed forward #{i} round")
#                 output_voxel, _ = model(batched_feats, batched_coords, batch_size)
#                 outputs = []
#                 for i in range(batch_size):
#                     outputs.append(output_voxel[i, :, batched_coords[i][:, 0], batched_coords[i][:, 1], batched_coords[i][:, 2]])
                
#                 outputs = torch.cat(outputs, dim=1).T
#                 batched_scores = F.softmax(outputs, dim=1)  # + IMPORTANT STEPS
#                 multiround_batched_scores.append(batched_scores)

#             # batch_size * point_count * class_count
#             multiround_batched_scores = torch.stack(multiround_batched_scores)

#             batched_labels = multiround_batched_scores.sum(dim=0).argmax(dim=1)  # Label
#             batched_scores = multiround_batched_scores.var(dim=0)  # Uncertainty

#             # Save labels and uncertainty for each scene
#             for scene_id in range(config["test_batch_size"]):
#                 print(f"Info: ---> processing #{scene_id} scene in the batch")
#                 selector = batched_coords[:, 0] == scene_id
#                 single_scene_labels = batched_labels[selector]
#                 single_scene_scores = batched_scores[selector]

#                 torch.save(single_scene_labels.cpu(), f"{config.unc_result_dir}/{names[scene_cnt].split('.')[0]}_predicted.obj")
#                 torch.save(single_scene_scores.cpu(), f"{config.unc_result_dir}/{names[scene_cnt].split('.')[0]}_unc.obj")

#                 scene_cnt += 1
