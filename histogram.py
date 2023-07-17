# %%
import h5py
from dataloader.semantickitti import SemanticKITTI
import yaml
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def save_sample(xyzr, label, idx, path='tmp/test.txt'):
    rgb = []
    with open(path, 'w') as f:
        for (x, y, z, _), l in zip(xyzr, label):
            r, g, b = color_map[l.item()]
            rgb.append(color_map[l.item()])
            # f.write(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}")
            print(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}", file=f)
    rgb = np.array(rgb)
    render(xyzr[:, :3], rgb, idx)
    
def render(xyz, rgb, idx):
    x=xyz[:,0]
    y=xyz[:,1]
    z=xyz[:,2]
    
    fig=plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x, y, z, c = rgb / 255.0, s = 4)
    ax.set_title(f'{idx}')
    
    plt.savefig(f"visualize/{idx}.png")

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
    nclass = 19
    unc_per_label = np.empty(nclass, dtype=np.float32)
    
    for i in range(nclass):
        unc_per_label[i] = []
    
    for i in tqdm(range(len(ds))):
        label_path = ds.label_paths[i]
        
        xyzr = ds.get_lidar(i)

        predicted_labels = hf[os.path.join(label_path, 'label')][()]
        uncertainty_scores = hf[os.path.join(label_path, 'unc')][()]
        
        for j in range(nclass):
            unc_per_label[j].extend(uncertainty_scores[predicted_labels == j])
            
    
       

# %%
