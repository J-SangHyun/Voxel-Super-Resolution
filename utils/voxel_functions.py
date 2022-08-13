# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage


def binvox2numpy(filename):
    with open(filename, 'rb') as fp:
        line = fp.readline().strip()
        assert line.startswith(b'#binvox')
        dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
        translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
        scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
        line = fp.readline()

        raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(bool)
        data = data.reshape(dims)
        data = np.transpose(data, (0, 2, 1))
        a, b, c = np.where(data == 1)
        model = np.zeros(dims)
        for x, y, z in zip(a, b, c):
            model[x, y, z] = 1
        model[ndimage.binary_fill_holes(model)] = 1
    return model


def voxel2mesh(voxels, threshold=0.0):
    top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    top_faces = [[1, 2, 4], [2, 3, 4]]

    bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    bottom_faces = [[2, 1, 4], [3, 2, 4]]

    left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    left_faces = [[1, 2, 4], [3, 1, 4]]

    right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    right_faces = [[2, 1, 4], [1, 3, 4]]

    front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    front_faces = [[2, 1, 4], [1, 3, 4]]

    back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    back_faces = [[1, 2, 4], [3, 1, 4]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim + 2, dim + 2, dim + 2))
    new_voxels[1:dim + 1, 1:dim + 1, 1:dim + 1] = voxels
    voxels = new_voxels

    scale = 0.01
    cube_dist_scale = 1.
    verts = []
    faces = []
    curr_vert = 0
    a, b, c = np.where(voxels > threshold)
    for i, j, k in zip(a, b, c):
        if voxels[i, j, k + 1] < threshold:
            verts.extend(scale * (top_verts + cube_dist_scale * np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(top_faces + curr_vert)
            curr_vert += len(top_verts)

        if voxels[i, j, k - 1] < threshold:
            verts.extend(scale * (bottom_verts + cube_dist_scale * np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(bottom_faces + curr_vert)
            curr_vert += len(bottom_verts)

        if voxels[i - 1, j, k] < threshold:
            verts.extend(scale * (left_verts + cube_dist_scale * np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(left_faces + curr_vert)
            curr_vert += len(left_verts)

        if voxels[i + 1, j, k] < threshold:
            verts.extend(scale * (right_verts + cube_dist_scale * np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(right_faces + curr_vert)
            curr_vert += len(right_verts)

        if voxels[i, j + 1, k] < threshold:
            verts.extend(scale * (front_verts + cube_dist_scale * np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(front_faces + curr_vert)
            curr_vert += len(front_verts)

        if voxels[i, j - 1, k] < threshold:
            verts.extend(scale * (back_verts + cube_dist_scale * np.array([[i - 1, j - 1, k - 1]])))
            faces.extend(back_faces + curr_vert)
            curr_vert += len(back_verts)

    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    with open(filename, 'w') as f:
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, threshold=0.4):
    verts, faces = voxel2mesh(pred, threshold)
    write_obj(filename, verts, faces)
