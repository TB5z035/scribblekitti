# %%
import yaml
from dataloader.semantickitti import SemanticKITTI

color_map = {
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

sample = dataset[0]

def save_sample(xyzr, label, path='/tmp/test.txt'):
    with open(path, 'w') as f:
        xyzr, label = sample
        for (x, y, z, _), l in zip(xyzr, label):
            r, g, b = color_map[l.item()]
            print(f"{x.item()} {y.item()} {z.item()} {r} {g} {b}", file=f)

if __name__ == '__main__':
    config_path = 'config/test.yaml'
    dataset_config_path = 'config/semantickitti.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))

    dataset = SemanticKITTI(split='train', config=config['dataset'])
    val_dataset = SemanticKITTI(split='valid', config=config['val_dataset'])

    sample = dataset[0]
    save_sample(*sample, path='/tmp/test.txt')
    val_sample = val_dataset[0]
    save_sample(*val_sample, path='/tmp/valtest.txt')
        

# %%
