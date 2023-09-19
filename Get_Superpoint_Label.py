# from src.transforms import Transform
from src.transforms.neighbors import knn_1
import numpy as np
import torch
from src.utils import sizes_to_pointers
from pgeof import pgeof
from src.data import Data
from sklearn.linear_model import RANSACRegressor
from src.transforms.partition import CutPursuitPartition, GridPartition
import matplotlib.pyplot as plt
from src.transforms import ConnectIsolated


def save_matrix_to_txt(matrix, filename):
    # Define the format string to separate elements with spaces
    format_str = ' '.join(['%s'] * matrix.shape[1])
    # Save the matrix to the text file
    np.savetxt(filename, matrix, fmt=format_str)

def get_points_with_rgb(points, groups):
    # Get the unique group labels
    unique_groups = np.unique(groups)

    # Generate a color map based on the number of unique groups
    num_colors = max(len(unique_groups), 20)
    if num_colors <= 20:
        cmap = plt.get_cmap('tab20b')
    else:
        cmap = plt.get_cmap('rainbow')

    colors = cmap(np.linspace(0, 1, num_colors))

    # Create an RGB matrix
    rgb_matrix = np.zeros((groups.shape[0], 3), dtype=np.uint8)

    # Assign colors to each group label
    for i, group in enumerate(unique_groups):
        indices = np.where(groups == group)[0]
        rgb_matrix[indices] = (colors[i-1][:3] * 255).astype(np.uint8)

    points_with_rgb = np.concatenate((points,rgb_matrix),axis=1)

    return points_with_rgb

def AddKeysTo_process_single_key(data, key, to, AddKeys_config):
    # Read existing features and the attribute of interest
    feat = getattr(data, key, None)
    x = getattr(data, to, None)

    # Skip if the attribute is None
    if feat is None:
        if AddKeys_config['strict']:
            raise Exception(f"Data should contain the attribute '{key}'")
        else:
            return data

    # Remove the attribute from the Data, if required
    if AddKeys_config['delete_after']:
        delattr(data, key)

    # In case Data has no features yet
    if x is None:
        if AddKeys_config['strict'] and data.num_nodes != feat.shape[0]:
            raise Exception(f"Data should contain the attribute '{to}'")
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)
        data[to] = feat
        return data

    # Make sure shapes match
    if x.shape[0] != feat.shape[0]:
        raise Exception(
            f"The tensors '{to}' and '{key}' can't be concatenated, "
            f"'{to}': {x.shape[0]}, '{key}': {feat.shape[0]}")

    # Concatenate x and feat
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if feat.dim() == 1:
        feat = feat.unsqueeze(-1)
    data[to] = torch.cat([x, feat], dim=-1)

    return data

def AddKeyTo_process(data, AddKeys_config):
    if AddKeys_config['keys'] is None or len(AddKeys_config['keys']) == 0:
        return data

    for key in AddKeys_config['keys']:
        data = AddKeysTo_process_single_key(data, key, AddKeys_config['to'],AddKeys_config)

    return data


def get_lidar( path):
    lidar = np.fromfile(path, dtype=np.float32)
    lidar = torch.from_numpy(lidar.reshape((-1, 4)))
    return lidar[:,:3]


def GroundElevation_process(data, GroundElevation_config):
    # Recover the point positions
    pos = data.pos.cpu().numpy()

    # To avoid capturing high above-ground flat structures, we only
    # keep points which are within `threshold` of the lowest point.
    idx_low = np.where(pos[:, 2] - pos[:, 2].min() < GroundElevation_config['threshold'])[0]

    # Search the ground plane using RANSAC
    ransac = RANSACRegressor(min_samples=0.5,random_state=0, residual_threshold=1e-3).fit(pos[idx_low, :2], pos[idx_low, 2])

    # Compute the pointwise elevation as the distance to the plane
    # and scale it
    h = pos[:, 2] - ransac.predict(pos[:, :2])
    h = h / GroundElevation_config['scale']

    # Save in Data attribute `elevation`
    data.elevation = torch.from_numpy(h).to(data.device).view(-1, 1)

    return data
data = Data()
data.pos = get_lidar('/data22/tb5zhh/datasets/SemanticKITTI/sequences/00/velodyne/000000.bin')
# data = torch.from_numpy(data)
config = {'keys':['linearity','planarity','scattering','verticality','curvature','length','surface','volume','normal'],'AdjacencyGraph':{'k':10,'w':1}}

##################################################### KNN #####################################################
neighbors, distances = knn_1(data.pos[:,:3], k=45, r_max=2, oversample=False,self_is_neighbor=False, verbose=False)
# neighbors, distances = knn_1(data[:,:3], k=config['knn']['k'], r_max=config['knn']['r_max'], oversample=False,self_is_neighbor=False, verbose=False)
data.neighbor_index = neighbors
data.neighbor_distance = distances


##################################################### GroundElevation #####################################################
GroundElevation_config = {'threshold': 5,'scale':20}
data = GroundElevation_process(data,GroundElevation_config)


##################################################### Point-Features #####################################################
assert data.has_neighbors, \
            "Data is expected to have a 'neighbor_index' attribute"
assert data.num_nodes < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"
assert data.neighbor_index.max() < np.iinfo(np.uint32).max, \
            "Too high 'neighbor_index' indices for `uint32` indices"
# get point-wise density feature
dmax = data.neighbor_distance.max(dim=1).values
k = data.neighbor_index.ge(0).sum(dim=1)
data.density = (k / dmax ** 2).view(-1, 1)


# get geometric features 
# device = data.pos.device
# xyz = data[:,:3].cpu().numpy()
device = data.pos.device
xyz = data.pos.cpu().numpy()
nn = torch.cat(
    (torch.arange(xyz.shape[0]).view(-1, 1), data.neighbor_index),
    dim=1)
k = nn.shape[1]

# Check for missing neighbors (indicated by -1 indices)
n_missing = (nn < 0).sum(dim=1)
if (n_missing > 0).any():
    sizes = k - n_missing
    nn = nn[nn >= 0]
    nn_ptr = sizes_to_pointers(sizes.cpu())
else:
    nn = nn.flatten().cpu()
    nn_ptr = torch.arange(xyz.shape[0] + 1) * k
nn = nn.numpy().astype('uint32')
nn_ptr = nn_ptr.numpy().astype('uint32')

# Make sure array are contiguous before moving to C++
xyz = np.ascontiguousarray(xyz)
nn = np.ascontiguousarray(nn)
nn_ptr = np.ascontiguousarray(nn_ptr)
# C++ geometric features computation on CPU
# f = pgeof(
#     xyz, nn, nn_ptr, k_min=config['pgeof']['k_min'], k_step=config['pgeof']['k_step'],
#     k_min_search=config['pgeof']['k_min_search'], verbose=False)
f = pgeof(
    xyz, nn, nn_ptr, k_min=1, k_step=-1,
    k_min_search=25, verbose=False)
f = torch.from_numpy(f.astype('float32'))
if 'linearity' in config['keys']:
    # data.linearity = f[:, 0].view(-1, 1).to(device)
    data.linearity = f[:, 0].view(-1, 1)

if 'planarity' in config['keys']:
    # data.planarity = f[:, 1].view(-1, 1).to(device)
    data.planarity = f[:, 1].view(-1, 1)

if 'scattering' in config['keys']:
    # data.scattering = f[:, 2].view(-1, 1).to(device)
    data.scattering = f[:, 2].view(-1, 1)

# Heuristic to increase importance of verticality in
# partition
if 'verticality' in config['keys']:
    # data.verticality = f[:, 3].view(-1, 1).to(device)
    data.verticality = f[:, 3].view(-1, 1)
    data.verticality *= 2

if 'curvature' in config['keys']:
    # data.curvature = f[:, 10].view(-1, 1).to(device)
    data.curvature = f[:, 10].view(-1, 1)

if 'length' in config['keys']:
    # data.length = f[:, 7].view(-1, 1).to(device)
    data.length = f[:, 7].view(-1, 1)

if 'surface' in config['keys']:
    # data.surface = f[:, 8].view(-1, 1).to(device)
    data.surface = f[:, 8].view(-1, 1)

if 'volume' in config['keys']:
    # data.volume = f[:, 9].view(-1, 1).to(device)
    data.volume = f[:, 9].view(-1, 1)

# As a way to "stabilize" the normals' orientation, we
# choose to express them as oriented in the z+ half-space
if 'normal' in config['keys']:
    # data.normal = f[:, 4:7].view(-1, 3).to(device)
    data.normal = f[:, 4:7].view(-1, 3)
    data.normal[data.normal[:, 2] < 0] *= -1


##################################################### AdjacencyGraph #####################################################

AdjacencyGraph_k = config['AdjacencyGraph']['k']
AdjacencyGraph_w = config['AdjacencyGraph']['w']

# Compute source and target indices based on neighbors
source = torch.arange(
    data.num_nodes, device=data.device).repeat_interleave(AdjacencyGraph_k)
target = data.neighbor_index[:, :AdjacencyGraph_k].flatten()

# Account for -1 neighbors and delete corresponding edges
mask = target >= 0
source = source[mask]
target = target[mask]

# Save edges and edge features in data
data.edge_index = torch.stack((source, target))
if AdjacencyGraph_w > 0:
    # Recover the neighbor distances and apply the masking
    distances_After_Ad = distances[:, :AdjacencyGraph_k].flatten()[mask]
    data.edge_attr = 1 / (AdjacencyGraph_w + distances_After_Ad / distances_After_Ad.mean())
else:
    data.edge_attr = torch.ones_like(source, dtype=torch.float)

# ConnectIsolated
ConnectIsolated_config = {'k':1}
ConnectIsolated_Transforms = ConnectIsolated(k=ConnectIsolated_config['k'])

# AddKeysTo
AddKeys_config = {'keys':['linearity','planarity','scattering','verticality','elevation'],'to':'x','delete_after':False,'strict':True}
data = AddKeyTo_process(data=data, AddKeys_config=AddKeys_config)

# CutPursuitPartition
CutPursuitPartition_config = {
    'pcp_regularization': [0.1, 0.2, 0.6],
    'pcp_spatial_weight': [1, 1e-1, 1e-2],
    'pcp_cutoff': [10, 30, 100],
    'pcp_k_adjacency': 10,
    'pcp_w_adjacency': 1,
    'pcp_iterations': 15,
    'parallel': True,
    'verbose': False}
CutPursuitPartition_Transforms = CutPursuitPartition(regularization=CutPursuitPartition_config['pcp_regularization'],
                                                     spatial_weight=CutPursuitPartition_config['pcp_spatial_weight'],
                                                     cutoff = CutPursuitPartition_config['pcp_cutoff'],
                                                     parallel=CutPursuitPartition_config['parallel'],
                                                     iterations=CutPursuitPartition_config['pcp_iterations'],
                                                     k_adjacency=CutPursuitPartition_config['pcp_k_adjacency'],
                                                     verbose=CutPursuitPartition_config['verbose'])
nag = CutPursuitPartition_Transforms(data)
super_index_0 = nag[0].super_index
# points_with_rgb = get_points_with_rgb(data.pos,super_index_0)
# save_matrix_to_txt(points_with_rgb,'level_1_super_index_points.txt')
# points_with_rgb = get_points_with_rgb(data.pos,(nag[1].super_index)[nag[0].super_index])
# save_matrix_to_txt(points_with_rgb,'level_2_super_index_points.txt')
# points_with_rgb = get_points_with_rgb(data.pos,nag[2].super_index[(nag[1].super_index)[nag[0].super_index]])
# save_matrix_to_txt(points_with_rgb,'level_3_super_index_points.txt')
# points_with_rgb = get_points_with_rgb(data.pos,nag[3].super_index[nag[2].super_index[(nag[1].super_index)[nag[0].super_index]]])
# save_matrix_to_txt(points_with_rgb,'level_4_super_index_points.txt')
group_level_3 = nag[2].super_index[(nag[1].super_index)[nag[0].super_index]]




print(neighbors.shape)
print(distances.shape)
print(data[:,:3].shape)

a = 1