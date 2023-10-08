import numpy as np
import torch

import os

from .csbdeep_plot_utils import plot_some

def arr_info(arr, name:str='Array'):
    assert type(arr) in [np.ndarray, torch.Tensor], 'Invalid array type.'
    print('==========')
    print(f'[{name}]:\n'
          f'shape: {arr.shape}\n'
          f'dtype: {arr.dtype}\n'
          f'max: {arr.max()}\n'
          f'min: {arr.min()}\n'
          f'mean: {arr.mean()}\n'
          f'std: {arr.std()}\n'
          f'sum: {arr.sum()}')
    if type(arr) == torch.Tensor:
        print(f'device: {arr.device}')
    print('==========')
def plot_mip(data:np.ndarray, figsize=(10,4), dpi=300, suptitle:str=None):
    assert len(data.shape)==3, 'The dimension of input array should be 3.'
    show_list = []
    for i in range(3):
        show_list.append(np.max(data, axis=i))
    plot_some(show_list, figsize=figsize, dpi=dpi, suptitle=suptitle, title_list=[['[ ][1][2]', '[0][ ][2]', '[0][1][ ]']])

def gene_gaussian_kernel_3d(sigma_x, sigma_y, sigma_z, size):
    kernel = np.zeros((size, size, size), dtype=np.float32)

    center_x = size // 2
    center_y = size // 2
    center_z = size // 2

    for x in range(size):
        for y in range(size):
            for z in range(size):
                exponent = (
                    ((x - center_x) ** 2) / (2 * sigma_x ** 2) +
                    ((y - center_y) ** 2) / (2 * sigma_y ** 2) +
                    ((z - center_z) ** 2) / (2 * sigma_z ** 2)
                )
                kernel[x, y, z] = np.exp(-exponent) / (2 * np.pi * sigma_x * sigma_y * sigma_z)

    kernel /= kernel.sum()
    return kernel

def gene_gaussian_kernel_2d(sigma_x, sigma_y, size):
    kernel = np.zeros((size, size), dtype=np.float32)

    center_x = size // 2
    center_y = size // 2

    for x in range(size):
        for y in range(size):
            exponent = (
                ((x - center_x) ** 2) / (2 * sigma_x ** 2) +
                ((y - center_y) ** 2) / (2 * sigma_y ** 2)
            )
            kernel[x, y] = np.exp(-exponent) / (2 * np.pi * sigma_x * sigma_y)

    kernel /= kernel.sum()
    return kernel

def normalize(arr, mode: str='min_max', *, max_v=None, min_v=0):
    if mode == 'min_max':
        norm = lambda t:(t-t.min())/(t.max()-t.min())
    elif mode == 'z_score':
        norm = lambda t:(t-t.mean())/(t.std())
    elif mode == 'scale':
        norm = lambda t:(t-min_v)/(max_v-min_v)
    else:
        Exception('Invalid mode.')
    return norm(arr)

def check_dir(path, *, mode:str='r'):
    if os.path.exists(path):
            print(f'the directory already exists: ["{path}"]')
    else:
        if mode == 'r':
            os.makedirs(path)
            print(f'the directory has been created: ["{path}"]')
        elif mode == 'a':
            os.mkdir(path)
            print(f'the directory has been created: ["{path}"]')
        else:
            print(f'the directory doesn\'t exist: ["{path}"]')

def tensor_to_ndarr(t: torch.Tensor):
    return t.detach().cpu().numpy()