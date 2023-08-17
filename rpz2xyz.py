import torch
import numpy as np

def cyl2cart(rpz):
    # convert rpz to xyz
    max_bound = np.array([50,3.1415926,2])
    min_bound = np.array([0,-3.1415926,-4])
    spatial_shape = np.array([480,360,32])
    drpz = (max_bound - min_bound) / (spatial_shape - 1)
    x = np.multiply(rpz[:, 1] * drpz[0], np.cos(rpz[:, 2] * drpz[1]))
    y = np.multiply(rpz[:, 1] * drpz[0], np.sin(rpz[:, 2] * drpz[1]))
    return np.stack((x, y, rpz[:, 3] * drpz[2]), axis=1)

if __name__ == "__main__":
    # unique_coords_b_transformed = np.loadtxt("b.txt")
    xyz_b = cyl2cart(unique_coords_b_transformed.cpu().numpy())
    full_b = np.pad(xyz_b, (0,3), 'constant', constant_values=(0))
    np.savetxt("b_mesh.txt", full_b, delimiter=",")
    # save as meshlab file
    # unique_coords_a = np.loadtxt("output.txt")
    xyz_a = cyl2cart(unique_coords_a.cpu().numpy())
    full_a = np.pad(xyz_a, (0,3), 'constant', constant_values=(0))
    np.savetxt("a_mesh.txt", full_a, delimiter=",")