# cyh: 用于看看semantickitti数据集的内容
import numpy as np
from plyfile import PlyData

filepath = "/DATA_EDS/tb5zhh/legacy/3d_scene_understand/SUField/results_0223/generate_datasets/20_fit_spec/train/scene0585_01.ply" 

plydata = PlyData.read(filepath)
data = plydata.elements[0].data
coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
labels = np.array(data['label'], dtype=np.int32)
import IPython; IPython.embed()