import numpy as np

def read_npy(path):
    mean_size_array = np.load(path)['arr_0']
    pass

if __name__ == '__main__':
    path = 'scannet_means.npz'
    read_npy(path)