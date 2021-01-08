"""
Description:
    - System for storing game experience
    - Can read and write experience blocks
    - Can randomly sample experience blocks
"""

import numpy as np
import torch
import pickle
from collections import deque

class ExperienceReplay:
    """
    Stores data collected from rollouts. It is important to update the
    h values after each update. Otherwise stale hs can infect the
    training.
    """
    def __init__(self, intl_data=None, max_size=20000, n_runs=16,
                                                       n_tsteps=32):
        """
        max_size: int
            the maximum number of time steps to store
        n_runs: int
            the number of 
        """
        self.n_runs = n_runs
        self.n_tsteps = n_tsteps
        self.max_size = max_size//self.n_runs
        self.data = {
            "obsrs":     None,
            "rews":      None,
            "hs":        None,
            "count_idxs":None,
            "longs":     None,
            #"color_idxs": idx 0
            #"shape_idxs": idx 1
            #"starts":     idx 2
            #"dones":      idx 3
            #"resets":     idx 4
        }
        # Fill main dict with with intl_data
        if intl_data is not None:
            self.add_new_data(intl_data)

    def add_data(self, new_data):
        """
                    R = self.n_runs
                    T = self.n_tsteps

            new_data: dict
                keys:
                    "obsrs":      torch float tensor (R*T, C,H,W)
                    "rews":       torch float tensor (R*T,)
                    "hs":         torch float tensor (R*T, E)
                    "count_idxs": torch long tensor (R*T,)
                    "longs":      torch long tensor (R*T,6)
                        #"color_idxs": idx 0
                        #"shape_idxs": idx 1
                        #"starts":     idx 2
                        #"dones":      idx 3
                        #"resets":     idx 4
        """
        # Append new data
        for k in self.data.keys():
            end_shape = [-1]
            if len(new_data[k].shape) > 1:
                end_shape = new_data[k].shape[1:]
            new_data[k] = new_data[k].reshape(self.n_runs,
                                              self.n_tsteps,
                                              *end_shape)
            new_data[k] = new_data[k].cpu().data.squeeze().numpy()
            if self.data[k] is not None:
                arr = [self.data[k],new_data[k]]
                self.data[k] = np.concatenate(arr, axis=1)
            else:
                self.data[k] = new_data[k]
            if len(self.data[k]) > self.max_len:
                self.data[k] = self.data[k]

    def get_data(self, idxs, horizon=9):
        """
        Returns a batch of data sequences of length horizon. The argued
        idxs indicate a batch of sequences starting with the data point
        of the argued indexes

        idxs: ndarray (B,)
            each index corresponds to a data sequence starting with
            the datapoint at that index in the data arrays 
        horizon: int
            the length of the data sequences
        """
        idxs = idxs.cpu().data.numpy()
        sample = dict()
        obs_seq = rolling_window(self.data['obsrs'], horizon+1)[idxs]
        sample['obs_seq'] = torch.FloatTensor(obs_seq)
        rew_seq = rolling_window(self.data['rews'], horizon+1)[idxs]
        sample['rew_seq'] = torch.FloatTensor(rew_seq)
        h_seq = rolling_window(self.data['hs'], horizon+1)[idxs]
        sample['h_seq'] = torch.FloatTensor(h_seq)
        count_seq = rolling_window(self.data['count_idxs'],
                                   horizon+1)[idxs]
        sample['count_seq'] = torch.LongTensor(count_seq)

        #"color_idxs": idx 0
        #"shape_idxs": idx 1
        #"starts":     idx 2
        #"dones":      idx 3
        #"resets":     idx 4
        long_seq = rolling_window(self.data['longs'],horizon+1)[idxs]
        long_seq = torch.LongTensor(long_seq)
        sample['color_seq'] = long_seq[:,:,0]
        sample['shape_seq'] = long_seq[:,:,1]
        sample['done_seq'] = long_seq[:,:,3]
        sample['reset_seq'] = long_seq[:,:,4]
        return sample # shapes (B,S,...)

    def update_hs(self, idxs, new_hs):
        """
        To avoid stale hidden states, we want to update the h vectors
        after each model update. This function should be called after
        each update using the same indices that were used for `get_data`

        idxs: torch Long Tensor (N,)
        new_hs: torch Float Tensor (N,S,H)
            the new h vectors. there should be a sequence of h values
            for each idx
        """
        horizon = new_hs.shape[1]
        idxs = idxs.cpu().data.numpy()
        new_hs = new_hs.cpu().data.numpy()
        for i in range(len(new_hs)):
            self.data[hs][idxs[i]:idxs[i]+horizon] = new_hs[i]

    def __len__(self):
        if self.data['dones'] is None:
            return 0
        return len(self.data['dones'])

    def load(self, save_name):
        with open(save_name, 'rb') as f:
            data = pickle.load(f)
        self.add_new_data(data)

    def save(self, save_name):
        with open(save_name, 'wb') as f:
            pickle.dump(self.data, f)

def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension.

    Taken from deepretina package (https://github.com/baccuslab/deep-retina/)

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if time_axis == 0:
        if type(array) == type(np.array([])):
            array = array.T
        elif len(array.shape) >= 2:
            l = list([i for i in range(len(array.shape))])
            array = array.transpose(*reversed(l))

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window <= array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - (window-1), window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr

