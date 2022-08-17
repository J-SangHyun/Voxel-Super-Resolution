# -*- coding: utf-8 -*-
import os
import platform
import subprocess
from tqdm import tqdm
from pathlib import Path


def preprocess(dataset_name, grid):
    project_path = Path(os.path.dirname(__file__))
    dataset_path = project_path / 'dataset'
    train_dir = {
        'ModelNet10': 'ModelNet10/*/train',
        'ModelNet40': 'ModelNet40/*/train',
        'examples': 'InvalidPath'
    }
    test_dir = {
        'ModelNet10': 'ModelNet10/*/test',
        'ModelNet40': 'ModelNet40/*/test',
        'examples': 'examples'
    }
    extension = {
        'ModelNet10': 'off',
        'ModelNet40': 'off',
        'examples': 'obj'
    }
    assert 512 >= grid

    # load meshes
    train_mesh_files = list(dataset_path.glob(train_dir[dataset_name] + f'/*.{extension[dataset_name]}'))
    test_mesh_files = list(dataset_path.glob(test_dir[dataset_name] + f'/*.{extension[dataset_name]}'))
    print(f'{len(train_mesh_files)} Train Files, {len(test_mesh_files)} Test Files Found.')

    # voxelize meshes using binvox
    current_os = platform.system()
    if current_os == 'Windows':
        binvox_path = project_path / 'utils' / 'binvox.exe'
    elif current_os == 'Linux':
        binvox_path = project_path / 'utils' / 'binvox'
    else:
        raise OSError(f'Current OS ({current_os}) is not supported. Only supports Windows and Linux.')

    def remove_binvox_files():
        train_binvox_files = list(dataset_path.glob(train_dir[dataset_name] + f'/*.binvox'))
        test_binvox_files = list(dataset_path.glob(test_dir[dataset_name] + f'/*.binvox'))
        print(f'Removing binvox files... ')
        for binvox_files in tqdm(train_binvox_files + test_binvox_files):
            Path.unlink(binvox_files)

    def mesh_to_binvox(grid_size):
        print(f'Preprocessing... (Mesh -> Binvox) | Grid Size = {grid_size}')
        for mesh_file in tqdm(train_mesh_files + test_mesh_files):
            args = [f'{binvox_path}', f'{mesh_file}', '-d', f'{grid_size}', '-pb', '-cb', '-c', '-dc', '-aw', '-e']
            subprocess.run(args, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    voxel_path = project_path / 'voxelized'
    Path.mkdir(voxel_path) if not Path.exists(voxel_path) else None

    def move_binvox_files(grid_size):
        voxel_grid_path = voxel_path / f'{grid_size}'
        Path.mkdir(voxel_grid_path) if not Path.exists(voxel_grid_path) else None
        train_binvox_files = list(dataset_path.glob(train_dir[dataset_name] + f'/*.binvox'))
        test_binvox_files = list(dataset_path.glob(test_dir[dataset_name] + f'/*.binvox'))
        for binvox_file in tqdm(train_binvox_files + test_binvox_files):
            path = binvox_file
            parents_paths = [binvox_file.name]
            while path.parent != project_path / 'dataset':
                path = path.parent
                parents_paths.insert(0, path.name)
            path = voxel_grid_path
            for p in parents_paths[:-1]:
                path = path / p
            Path.mkdir(path, parents=True) if not Path.exists(path) else None
            Path.unlink(path / parents_paths[-1]) if Path.exists(path / parents_paths[-1]) else None
            binvox_file.rename(path / parents_paths[-1])

    remove_binvox_files()
    mesh_to_binvox(grid)
    move_binvox_files(grid)
    remove_binvox_files()


if __name__ == '__main__':
    dataset_names = ['ModelNet10', 'ModelNet40', 'examples']
    print('------- Dataset to Voxelize --------')
    for i in range(len(dataset_names)):
        print(f'{i}. {dataset_names[i]}')
    dataset_name = dataset_names[int(input('Choose Dataset: '))]

    grid_sizes = [16, 32, 64, 128, 256]
    for grid in grid_sizes:
        preprocess(dataset_name, grid)
