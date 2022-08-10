# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm


def load_voxels_from_dataset(dataset_name, voxel_size=1/49):
    # train & test path of dataset
    dataset_path = os.path.dirname(__file__)
    train_path = {
        'ModelNet10': 'ModelNet10/*/train/*.off',
        'ModelNet40': 'ModelNet40/*/train/*.off',
    }
    test_path = {
        'ModelNet10': 'ModelNet10/*/test/*.off',
        'ModelNet40': 'ModelNet40/*/test/*.off',
    }

    # check validity of model name
    assert dataset_name in train_path.keys()
    assert dataset_name in test_path.keys()

    # load meshes
    train_files = glob(os.path.join(dataset_path, train_path[dataset_name]))
    test_files = glob(os.path.join(dataset_path, test_path[dataset_name]))
    print(f'{len(train_files)} Train Files, {len(test_files)} Test Files Found.')

    # preprocess meshes
    print(f'Train meshes preprocessing')
    for train_f in tqdm(train_files):
        mesh = o3d.io.read_triangle_mesh(train_f)
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1, 0.706, 0])
        #o3d.visualization.draw_geometries([mesh], width=1280, height=720)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)
        #o3d.geometry.TriangleMesh.
        #voxel_grid.compute_vertex_normals()
        o3d.visualization.draw_geometries([voxel_grid], width=1280, height=720)

        voxels = voxel_grid.get_voxels()
        #print(voxels)
        indices = np.stack(list(vx.grid_index for vx in voxels))
        print(indices)
        break


if __name__ == '__main__':
    pcd = load_voxels_from_dataset('ModelNet10')
