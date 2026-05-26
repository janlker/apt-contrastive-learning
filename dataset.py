from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.dataset_util import *
from scipy.spatial import KDTree
from pathlib import Path
from sklearn.utils import shuffle
from math import ceil

class APT_basic(Dataset):
    def __init__(
        self,
        data_path,
        num_points,
        data_size,                     
        input_features=["x","y","z","element"],
        elements=None,
        phase_dict=None,
        partition="train",
        num_positives=1,
        close_by_compare=0,
        augmentation=False,
        rotation="so3",
        jitter=0,
        seed=1234,
        dissolve_n=True,
    ):
        self.data_path = data_path
        self.num_points = num_points
        self.data_size = data_size
        self.partition = partition
        self.num_positives = num_positives
        self.close_by_compare = close_by_compare
        self.augmentation = augmentation
        self.rotation = rotation
        self.jitter = jitter
        self.seed = seed
        test_set_seed = 3210

        if self.partition == "train":
            self.rng = np.random.default_rng(seed=self.seed) # data sampling
        if self.partition == "test":
            self.rng = np.random.default_rng(test_set_seed) # data sampling, fixed

        self.dpos, self.max_dist_estimate = get_data_raw(
            data_path=self.data_path,
            input_features=input_features,
            elements=elements,
            k=self.num_points,
            phase_dict=phase_dict,
            dissolve_n=dissolve_n
        )

        self.dpos = shuffle(self.dpos, random_state=111) 

        self.kd_tree = KDTree(self.dpos[:, 0:3], leafsize=15)
        
        # Static IDs 
        data_ids = self.rng.choice(self.dpos.shape[0], self.data_size, replace=False)
        dd, self.ii = self.kd_tree.query(self.dpos[data_ids, 0:3], k=self.num_points, workers=4)
        self.data = self.dpos[self.ii].copy()
        self.label =  self.data[:, 0, -1].copy()
        self.data = self.data[:, :, 0:-1]

        self.data[:, :, 0:3] = normalize_unit_sphere_fixed_batched(self.data[:, :, 0:3], self.max_dist_estimate).astype("float32")

    def __getitem__(self, item):

        if not hasattr(self, 'worker_rng_data'):
            self.worker_rng_data, self.worker_rng_cbc = get_worker_rngs(self.seed) # worker_rng_data only needed in APT_dynamic
            self.worker_rng_aug = np.random.default_rng() # If cbc should be random, use this 

        pointcloud = self.data[item].copy()
        pointcloud_label = self.label[item].copy()
        ii_anchor = self.ii[item].copy()

        if self.num_positives != 0:

            pointcloud_transformed = []
            for n in range(self.num_positives):
                if self.close_by_compare != 0 and self.partition == "train":
                    
                    max_idx =  max(2, int(self.num_points * self.close_by_compare) + 1)  # upper bound
                    cbc_candidates = ii_anchor[1:max_idx]
                    id_cbc = self.worker_rng_cbc.choice(cbc_candidates)
                    center_atom_cbc = self.dpos[id_cbc, 0:3]
                    dd, ii_cbc = self.kd_tree.query(center_atom_cbc, k=self.num_points, workers=1)

                    data = self.dpos[ii_cbc].copy()
                    pointcloud_cbc = data[:, 0:-1]
                    pointcloud_cbc[:, 0:3] = normalize_unit_sphere_fixed(pointcloud_cbc[:, 0:3], self.max_dist_estimate).astype("float32")

                    pointcloud_transformed.append(transform_data_single(pointcloud_cbc, rotate=self.rotation, jitter=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)) 
                else:
                    pointcloud_transformed.append(transform_data_single(pointcloud, rotate=self.rotation, jitter=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)) 

            if self.augmentation and self.partition == "train": 
                pointcloud = rotate_pointcloud(pointcloud, mode=self.rotation, rng=self.worker_rng_aug)
                if self.jitter != 0:
                    pointcloud = jitter_pointcloud(pointcloud, sigma=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)

            return pointcloud, pointcloud_transformed, pointcloud_label
            
        else: #supervised case
            if self.augmentation and self.partition == "train":
                pointcloud = rotate_pointcloud(pointcloud, mode=self.rotation, rng=self.worker_rng_aug)
                if self.jitter != 0:
                    pointcloud = jitter_pointcloud(pointcloud, sigma=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)

            return pointcloud, pointcloud_label

    def __len__(self):
        return self.data_size

def build_dataset(config, partition="train", supervision="supervised"):

    # Adjustments based on supervision and partition
    num_positives = 0 # creates positive environments, only needed for self_supervised + train
    augmentation = False # controls augmentation to the anchor, only needed for supervised + train
    close_by_compare = 0 # adjusts positive environments, only needed for self_supervised + train

    if partition == "train":
        if supervision == "supervised":
            augmentation = config.augmentation
        elif supervision == "self_supervised":
            num_positives = config.num_positives
            close_by_compare = config.close_by_compare

    if config.dataset_mode == "APT_basic":

        data_size = config.train_size if partition == "train" else config.test_size

        dataset_out = APT_basic(
            data_path=config.data_path,
            num_points=config.num_points,
            input_features=config.input_features,
            elements=config.elements,
            seed=config.seed,
            data_size=data_size,
            num_positives=num_positives, 
            partition=partition,
            phase_dict=config.phase_dict,
            close_by_compare=close_by_compare,
            augmentation=augmentation,
            rotation=config.rotation,
            jitter=config.jitter,
        )

    else:
        raise NotImplementedError(f"Dataset mode '{config.dataset_mode}' is not implemented yet.")


    return dataset_out


