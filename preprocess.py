# -*- coding: utf-8 -*-
import os
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    dataset_name = 'ModelNet10'
    project_path = os.path.dirname(__file__)
    dataset_path = os.path.join(project_path, 'dataset/')
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

    # binvox
    binvox_path = os.path.join(os.path.join(project_path, 'utils/'), 'binvox.exe')
    grid_size = 256
    for mesh_file in train_files[:1]:
        #os.system(f'cmd /k "{binvox_path}')
        os.system(f'cmd /k "{binvox_path} {mesh_file} -d {grid_size} -pb -cb -c -dc -aw -e"')

