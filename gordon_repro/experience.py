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
    def __init__(self, intl_data=None, max_size=20000):
        """
        max_size: int or None
            the maximum number of time steps to store. if None, no max
            is imposed
        """
        self.max_size = max_size if max_size is not None else np.inf
        self.data = {
            "obsrs":     None,
            "rews":      None,
            "fwd_hs":    None,
            "count_idxs":None,
            "longs":     None,
            #"color_idxs": idx 0
            #"shape_idxs": idx 1
            #"starts":     idx 2
            #"dones":      idx 3
            #"resets":     idx 4
        }
        self.data_lists = {k:[] for k in self.data.keys()}
        # Fill main dict with with intl_data
        if intl_data is not None:
            self.add_data(intl_data)

    def add_data(self, new_data):
        """
                    R = self.n_runs
                    T = self.n_tsteps

            new_data: dict
                keys:
                    "obsrs":      torch float tensor (B,C,H,W)
                    "rews":       torch float tensor (B,)
                    "fwd_hs":         torch float tensor (B,E)
                    "count_idxs": torch long tensor  (B,)
                    "longs":      torch long tensor  (B,5)
                        #"color_idxs": idx 0
                        #"shape_idxs": idx 1
                        #"starts":     idx 2
                        #"dones":      idx 3
                        #"resets":     idx 4
        """
        # Append new data
        for k in self.data.keys():
            new_data[k] = new_data[k].cpu().detach().data.squeeze()
            self.data_lists[k].append(new_data[k].clone())

    def cat_new_data(self):
        """
        This function must be called before using the data dict. It
        performs the concatenation of all new data with the old data
        and imposes the max size limit
        """
        for k in self.data.keys():
            if len(self.data_lists[k]) > 0:
                if self.data[k] is None:
                    self.data[k] = torch.cat(self.data_lists[k],dim=0)
                else:
                    arr = [self.data[k],*self.data_lists[k]]
                    self.data[k] = torch.cat(arr, dim=0)
                if len(self.data[k])>self.max_size:
                    self.data[k] = self.data[k][-self.max_size:]
                self.data_lists[k] = []

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

        S = horizon
        B = len(idxs)
        Returns:
            dict
                "obs_seq": float tensor (B,S,C,H,W)
                "rew_seq": float tensor (B,S)
                "h_seq": float tensor   (B,S,E)
                "done_seq":  long tensor (B,S)
                "start_seq": long tensor (B,S)
                "reset_seq": long tensor (B,S)
                "color_seq": long tensor (B,S)
                "shape_seq": long tensor (B,S)
                "count_seq": long tensor (B,S)
        """
        self.cat_new_data()
        sample = dict()
        sample['obs_seq'] = rolling_window(self.data['obsrs'],
                                           horizon)[idxs].clone()
        sample['rew_seq'] = rolling_window(self.data['rews'],
                                           horizon)[idxs].clone()
        sample['h_seq'] = rolling_window(self.data['fwd_hs'],
                                           horizon)[idxs].clone()
        sample['count_seq'] = rolling_window(self.data['count_idxs'],
                                            horizon)[idxs].long().clone()
        #"color_idxs": idx 0
        #"shape_idxs": idx 1
        #"starts":     idx 2
        #"dones":      idx 3
        #"resets":     idx 4
        long_seq = rolling_window(self.data['longs'],
                                  horizon)[idxs].long().clone()
        sample['color_seq'] = long_seq[:,:,0]
        sample['shape_seq'] = long_seq[:,:,1]
        sample['start_seq'] = long_seq[:,:,2]
        sample['done_seq'] =  long_seq[:,:,3]
        sample['reset_seq'] = long_seq[:,:,4]
        sample = {k:v.contiguous() for k,v in sample.items()}
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
        new_hs = new_hs.cpu().data
        for i in range(len(new_hs)):
            self.data["fwd_hs"][idxs[i]:idxs[i]+horizon] = new_hs[i]

    def __len__(self):
        list_len = 0
        if self.data_lists['obsrs'] is not None:
            list_len=np.sum([len(l) for l in self.data_lists['obsrs']])
            list_len = int(list_len)
        data_len = 0
        if self.data['obsrs'] is not None:
            data_len = len(self.data['obsrs'])
        return min(list_len + data_len, self.max_size)

    def load(self, save_name):
        with open(save_name, 'rb') as f:
            data = pickle.load(f)
        self.add_data(data)

    def save(self, save_name):
        with open(save_name, 'wb') as f:
            pickle.dump(self.data, f)

def rolling_window(array, window, axis=0, stride=1):
    """
    Make an ndarray with a rolling window of the last dimension

    Taken from deepretina package (https://github.com/baccuslab/deep-retina/)

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    axis : int, optional
        The axis of the temporal dimension, either 0 or -1
        (Default: 0)

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
    if isinstance(array, torch.Tensor):
        assert axis==0, "time axis must be 0 for torch tensors"
        arr = array.unfold(axis, window, stride)
        arange = torch.arange(len(arr.shape))
        if len(arange) > 2:
            return arr.permute(0,arange[-1],*arange[1:-1])
        return arr

    if stride != 1:
        s = "strides other than 1 are not implemented for ndarrays"
        raise NotImplemented(s)
    if axis == 0:
        array = array.T

    elif axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1\
                                                              (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape,
                                             strides=strides)

    if axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr
