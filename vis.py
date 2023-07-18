# %%
import h5py
from dataloader.semantickitti import SemanticKITTI
import yaml
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata        
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

UNC_RANGE = 19

color_map_unc = {
  0: [0, 255, 0],
  1: [51, 255, 51],
  2: [102, 255, 102],
  3: [153, 255, 153],
  4: [204, 255, 204],
  5: [255, 255, 204],
  6: [255, 255, 153],
  7: [255, 255, 102],
  8: [255, 255, 51],
  9: [255, 255, 0],
  10: [255, 229, 204],
  11: [255, 204, 153],
  12: [255, 178, 102],
  13: [255, 153, 51],
  14: [255, 128, 0],
  15: [255, 204, 204],
  16: [255, 153, 153],
  17: [255, 102, 102],
  18: [255, 51, 51],
  19: [255, 0, 0]
}

color_map_label = {
  0: [0,0,0],
  1: [245, 150, 100],
  2: [245, 230, 100],
  3: [150, 60, 30],
  4: [180, 30, 80],
  5: [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [255, 0, 255],
  10: [255, 150, 255],
  11: [75, 0, 75],
  12: [75, 0, 175],
  13: [0, 200, 255],
  14: [50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [150, 240, 255],
  19: [0, 0, 255]
}

# sample = dataset[0]

def vis_label(xyzr, label, path='tmp/test.txt'):
    with open(path, 'w') as f:
        for (x, y, z, _), l in zip(xyzr, label):
            r, g, b = color_map_label[l.item()]
            # f.write(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}")
            print(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}", file=f)

def get_idx(u):
    if u < 10e-15:
        return 0
    if u < 10e-12:
        return 1
    if u < 10e-9:
        return 2
    if u < 10e-7:
        return 3
    if u < 10e-6:
        return 4
    if u < 10e-5:
        return 5
    if u < 0.0001:
        return 6
    if u < 0.0005:
        return 7
    if u < 0.0010:
        return 8
    if u < 0.0015:
        return 9
    if u < 0.0020:
        return 10
    if u < 0.0025:
        return 11
    if u < 0.0030:
        return 12
    if u < 0.0035:
        return 13
    if u < 0.0040:
        return 14
    if u < 0.0045:
        return 15
    if u < 0.0050:
        return 16
    if u < 0.0055:
        return 17
    if u < 0.0070:
        return 18
    if u < 0.01:
        return 19
    if u < 0.02:
        return 20
    if u < 0.03:
        return 21
    if u < 0.04:
        return 22
    if u < 0.05:
        return 23
    return 24
        

def vis_unc(xyzr, unc, path='tmp/test.txt'):
    with open(path, 'w') as f:
        for (x, y, z, _), u in zip(xyzr, unc):
            idx = get_idx(u)
            # print(idx)
            if idx > UNC_RANGE:
                idx = UNC_RANGE
            r, g, b = color_map_unc[idx]
            # f.write(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}")
            print(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}", file=f)

if __name__ == '__main__':
    # config_path = 'config/test.yaml'
    # dataset_config_path = 'config/semantickitti.yaml'

    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # with open(dataset_config_path, 'r') as f:
    #     config['dataset'].update(yaml.safe_load(f))
    # with open(dataset_config_path, 'r') as f:
    #     config['val_dataset'].update(yaml.safe_load(f))

    hf = h5py.File("./inference/training_results.h5", "r")
    config = yaml.safe_load(open('config/unc_filter.yaml', 'r'))
    config['dataset'].update(yaml.safe_load(open('config/dataset/semantickitti.yaml', 'r')))
    ds = SemanticKITTI(split='train', config=config['dataset'])
    point_cloud = []

    for i in range(20):
        point_cloud.append([])

    for i in tqdm(range(len(ds))):
        if i < 10000:
            continue
        label_path = ds.label_paths[i]
        
        xyzr = ds.get_lidar(i)

        predicted_labels = hf[os.path.join(label_path, 'label')][()]
        uncertainty_scores = hf[os.path.join(label_path, 'unc')][()]
        uncertainty_scores = np.sum(uncertainty_scores, axis=1)

        for j in range(len(xyzr)):
            point_cloud[predicted_labels[j]].append(uncertainty_scores[j])
        
        print(np.max(uncertainty_scores), np.min(uncertainty_scores))
        vis_label(xyzr, predicted_labels, f"tmp/label{i}.txt")
        vis_unc(xyzr, uncertainty_scores, f"tmp/unc{i}.txt")
 
        plt.figure()
        uncertainty_scores = uncertainty_scores[uncertainty_scores <= 0.004]
        uncertainty_scores = uncertainty_scores[uncertainty_scores >= 10e-6]
        
        plt.hist(uncertainty_scores,bins=100)
        
        plt.title("data analyze")
        plt.xlabel("height")
        plt.ylabel("rate")
        plt.savefig(f"tmp/unc_hist{i}.png")
        
        if i == 10010:
            break
        
    for i in range(20):
        plt.figure()
        plt.hist(point_cloud[i], bins=600)
        plt.title("data analyze")
        plt.xlabel("height")
        plt.ylabel("rate")
        plt.savefig(f"labels/unc_hist{i}.png")

# %%
