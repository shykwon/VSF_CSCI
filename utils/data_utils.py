# Data loading utilities — aligned with VIDA protocol
# Reference: vida-vsf/VIDA/util.py

import numpy as np
import os
import math


class StandardScaler:
    """Standardize input data (VIDA-compatible)."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoaderM:
    """Custom data loader matching VIDA's DataLoaderM."""
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.num_nodes = xs.shape[2]
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()


def load_dataset(args, dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    """
    Load preprocessed dataset (VIDA-compatible).

    Expects {dataset_dir}/{train,val,test}.npz with keys 'x', 'y'.
    x shape: (num_samples, seq_in_len, num_nodes, input_dim)
    y shape: (num_samples, seq_out_len, num_nodes, output_dim)
    """
    data = {}
    total_num_nodes = None
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        print(f"Shape of {category} input = {data['x_' + category].shape}")
        total_num_nodes = data['x_' + category].shape[2]
        data['total_num_nodes'] = total_num_nodes

    if args.predefined_S:
        count = math.ceil(total_num_nodes * (args.predefined_S_frac / 100))
        oracle_idxs = np.random.choice(np.arange(total_num_nodes), size=count, replace=False)
        data['oracle_idxs'] = oracle_idxs
        for category in ['train', 'val', 'test']:
            data['x_' + category] = data['x_' + category][:, :, oracle_idxs, :]
            data['y_' + category] = data['y_' + category][:, :, oracle_idxs, :]

    # VIDA protocol: scaler from windowed train input (channel 0)
    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std()
    )
    # Normalize only x, keep y raw (VIDA protocol)
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data
