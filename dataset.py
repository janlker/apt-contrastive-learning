from torch.utils.data import Dataset
import numpy as np
from utils.dataset_util import *
from scipy.spatial import KDTree
from pathlib import Path
from sklearn.utils import shuffle
from math import ceil
from scipy.stats import truncnorm

class APT_basic(Dataset):
    def __init__(
        self,
        data_path,
        num_points,
        train_size,
        test_size,                     
        input_features=["x","y","z","element"],
        elements=None,
        phase_dict=None,
        partition="train",
        test_tol=1,
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
        self.partition = partition
        self.num_positives = num_positives
        self.close_by_compare = close_by_compare
        self.augmentation = augmentation
        self.rotation = rotation
        self.jitter = jitter
        self.seed = seed
        self.test_tol = test_tol
        test_set_seed = 3210

        # define rng for train/test sets
        if self.partition == "train":
            self.rng = np.random.default_rng(seed=self.seed)
            rng_test = np.random.default_rng(test_set_seed) 
            self.data_size = train_size
        if self.partition == "test":
            rng_test = np.random.default_rng(test_set_seed) 
            self.data_size = test_size

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
        test_ids = rng_test.choice(self.dpos.shape[0], test_size, replace=False)

        if self.partition == "test":
            data_ids = test_ids
            
        else: # self.partition == "train"
            if self.test_tol < 1:
                # Build test atom mask
                _, ii_test = self.kd_tree.query(self.dpos[test_ids, 0:3], k=self.num_points, workers=4)
                self.is_test_atom = np.zeros(self.dpos.shape[0], dtype=bool)
                self.is_test_atom[ii_test.reshape(-1)] = True
                self.allowed_overlap = int(self.test_tol * self.num_points)

                chunk_s = 2500 
                valid_train_ids_max = self.data_size
                
                rnd_id = self.rng.choice(self.dpos.shape[0], self.dpos.shape[0], replace=False)
                chunks = int(ceil(self.dpos.shape[0] / chunk_s))
                
                valid_train_ids = []
                id_break = 0
                id_break_max = 50

                for c in range(chunks):
                    sl = slice(c*chunk_s, min((c+1)*chunk_s, self.dpos.shape[0]))
                    centers_idx = rnd_id[sl]
                    
                    center_atom_check = self.dpos[centers_idx, 0:3]
                    _, ii = self.kd_tree.query(center_atom_check, k=self.num_points, workers=-1)
                    
                    counts = self.is_test_atom[ii].sum(axis=1)
                    valid_mask = counts <= self.allowed_overlap
                    
                    valid_train_ids.extend(centers_idx[valid_mask].tolist())
                    
                    if len(valid_train_ids) >= valid_train_ids_max:
                        break
                    if id_break == id_break_max:
                        break
                    id_break += 1

                data_ids = np.array(valid_train_ids[:self.data_size])
                if len(data_ids) < self.data_size:
                    print(f"Warning: Only found {len(data_ids)} valid train IDs!")
                    
            else: # test_tol == 1
                data_ids = self.rng.choice(self.dpos.shape[0], self.data_size, replace=False)
        
        # generate dataset based on data ids
        dd, self.ii = self.kd_tree.query(self.dpos[data_ids, 0:3], k=self.num_points, workers=4)
        self.data = self.dpos[self.ii].copy()
        self.label =  self.data[:, 0, -1].copy()
        self.data = self.data[:, :, 0:-1]

        self.data[:, :, 0:3] = normalize_unit_sphere_fixed_batched(self.data[:, :, 0:3], self.max_dist_estimate).astype("float32")

    def __getitem__(self, item):

        if not hasattr(self, 'worker_rng_data'):
            self.worker_rng_data, self.worker_rng_cbc = get_worker_rngs(self.seed) # worker_rng_data only needed in APT_dynamic
            self.worker_rng_aug = np.random.default_rng() 

        pointcloud = self.data[item].copy()
        pointcloud_label = self.label[item].copy()
        ii_anchor = self.ii[item].copy()

        if self.num_positives != 0:

            pointcloud_transformed = []
            for n in range(self.num_positives):
                if self.close_by_compare !=0  and self.partition == "train": 

                    max_idx =  max(2, int(self.num_points * self.close_by_compare) + 1)  # upper bound
                    cbc_candidates = ii_anchor[1:max_idx]                          

                    valid_cbc = False # valid_cbc are atoms that also pass the test_tol test
                    for id_cbc in self.worker_rng_cbc.permutation(cbc_candidates): 
                        center_atom_cbc =  self.dpos[id_cbc, 0:3]
                        dd, ii_cbc = self.kd_tree.query(center_atom_cbc, k=self.num_points, workers=1)
                        if self.test_tol < 1:             # check if overlap ok
                            overlap = self.is_test_atom[ii_cbc].sum()
                            if overlap > self.allowed_overlap:
                                continue                 # reject & try next in list
                            else:
                                valid_cbc = True
                                break
                        else:
                            valid_cbc = True
                            break

                    if valid_cbc == False: # use anchor as fallback
                        ii_cbc = ii_anchor.copy()
                        id_cbc = ii_anchor[0]

                    data = self.dpos[ii_cbc].copy()
                    pointcloud_cbc = data[:,0:-1] # remove label indicator

                    pointcloud_cbc[:,0:3] = normalize_unit_sphere_fixed(pointcloud_cbc[:,0:3], self.max_dist_estimate).astype("float32")

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


class APT_dynamic(Dataset):
    def __init__(
        self,
        data_path,
        log_path,
        num_points=1024,
        input_features=["x","y","z","element"],
        elements=None,
        phase_identifier="VAlN",
        phase_dict={"Dummy":0},
        dissolve_n=True,
        seed=1234,
        train_size=5000,
        test_size=3000,
        num_positives=1,
        partition="train",
        dist_to_surface=0,
        close_by_compare=0,
        jitter=0,
        rotation="so3",
        test_tol=1,
        augmentation=False,
        sim_data=False,
        gaussian_noise=0,
        det_efficiency=1,
        isotropic_noise=False,
        epochs=100,
    ):
        self.num_points = num_points
        self.elements = elements
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.num_positives = num_positives
        self.partition = partition
        self.dist_to_surface = float(dist_to_surface)
        self.close_by_compare = close_by_compare
        self.test_tol = test_tol
        self.augmentation = augmentation
        self.jitter = jitter
        self.rotation = rotation
        self.dissolve_n = dissolve_n
        self.phase_dict = phase_dict


        # load APT data  and get estimate of density (based on selected elements)
        self.dpos, self.max_dist_estimate = get_data_raw(
            data_path=data_path,
            input_features=input_features,
            elements=elements,
            k=self.num_points,
            phase_dict=phase_dict,
            dissolve_n=dissolve_n)
        
        self.dpos = shuffle(self.dpos, random_state=111) 
        self.id_list_in_dpos = None  # id_list_in_dpos used for test set tracking (needed for test_tol < 1).
        test_set_seed = 3210

        # restrict atoms that may serve as center atoms based on their distance to estimated sample surface
        if self.dist_to_surface !=0:

            print(f"Ignore atoms with distance of ~{self.dist_to_surface} to surface as query atoms")

            if sim_data: # sim data assumes a rectangular shape, a min/max box check is enough
                coord_min = np.min(self.dpos[:,0:3], axis=0)
                coord_max = np.max(self.dpos[:,0:3], axis=0)
                self.dpos_query = self.dpos

                avoid_atoms = np.zeros(len(self.dpos_query), dtype=bool)
                for i in [0, 1, 2]:
                    avoid_atoms |= (
                        (self.dpos_query[:, i] <= coord_min[i] + self.dist_to_surface) |
                        (self.dpos_query[:, i] >= coord_max[i] - self.dist_to_surface)
                    )
                self.dpos_query = self.dpos[~avoid_atoms]

            else: # for experimental data

                # surface approximation based on alpha shapes. This a rough approximation based on a random sampling of the APT data
                if dissolve_n:
                    avoid_atoms_path = log_path + phase_identifier + "_dist_surface_" + str(self.dist_to_surface) +  ".npy"
                else:
                    avoid_atoms_path = log_path + phase_identifier + "_dist_surface_" + str(self.dist_to_surface) +  "_dissolve_false.npy"

                avoid_atoms_path = Path(avoid_atoms_path)
                if avoid_atoms_path.is_file(): # reuse identified surface atoms to ensure consistency and reduce computation time
                    avoid_atoms = np.load(avoid_atoms_path) 
                    print("Loading saved surface atoms file")
                    if avoid_atoms.shape[0] != self.dpos.shape[0]:
                        avoid_atoms = get_surface_points_alpha_shape(self.dpos, initial_alpha=2.5, mesh_until_watertight = True, dist_to_surface=self.dist_to_surface, return_bool=True)
                        print("Redo of surface detection. Saved surface atom file does not match current data length, please check. File not saved")
                else:
                    avoid_atoms = get_surface_points_alpha_shape(self.dpos, initial_alpha=2.5, mesh_until_watertight = True, dist_to_surface=self.dist_to_surface, return_bool=True)
                    np.save(avoid_atoms_path,  np.squeeze(avoid_atoms))
                    print("Save surface detection file")

                avoid_atoms = np.squeeze(avoid_atoms)
                self.dpos_query = self.dpos[~avoid_atoms]

        else: # no surface restriction, every atom may be a center atom.
            self.dpos_query = self.dpos
            avoid_atoms = np.zeros(self.dpos.shape[0], dtype=bool) #dummy
            print("No surface atom removal, use all atoms as query atoms")

        self.dpos_query_size = self.dpos_query.shape[0]
        self.idx_query_to_dpos = np.flatnonzero(~avoid_atoms).astype(int)
        self.kd_tree = KDTree(self.dpos[:, 0:3], leafsize=15)

        # train rng
        self.rng = np.random.default_rng(seed=self.seed)

        # Test data is needed for both partitions (if test_tol is < 1): for test set and to avoid placement of train set in their proximity
        if self.test_tol < 1: 
            self.rng_test = np.random.default_rng(seed=test_set_seed) 

            self.id_list = self.rng_test.choice(self.dpos_query_size , self.test_size, replace=False) # dpos_query space
            self.id_list_in_dpos = self.idx_query_to_dpos[self.id_list].copy() # dpos space. Test ids of center atoms for test set (dpos reference)

            test_centers = self.dpos[self.id_list_in_dpos, 0:3].copy()
            dd, self.ii_test = self.kd_tree.query(test_centers, k=self.num_points, workers=4) # save ids from test set to ensure to later check again train set

            # union of all test set atoms, as a dpos space boolean mask.
            self.is_test_atom = np.zeros(self.dpos.shape[0], dtype=bool)
            self.is_test_atom[self.ii_test.reshape(-1)] = True  
            self.allowed_overlap = int(self.test_tol * self.num_points)  # e.g., 0.25 * k
            print("Allowed overlap is: " + str(self.allowed_overlap))

        elif self.partition == "test" and self.test_tol == 1: # no leakage control
            self.rng_test = np.random.default_rng(seed=test_set_seed)
            self.id_list = self.rng_test.choice(self.dpos_query_size, self.test_size, replace=False)
            self.id_list_in_dpos = self.idx_query_to_dpos[self.id_list].copy()   
            center_atom = self.dpos[self.id_list_in_dpos, 0:3]                   
            dd, self.ii_test = self.kd_tree.query(center_atom, k=self.num_points, workers=4)


        # Simulated/artificial APT data => apply detector efficiency and noise
        # Applied AFTER test center atom selection (so the same atoms remain centers)
        # but BEFORE training-pool selection (so removal does not bias it).
        if sim_data:

            if det_efficiency != 1:

                dpos_init_size = self.dpos.shape[0]
                target_ids_size = int(dpos_init_size*det_efficiency) # 

                if self.id_list_in_dpos is not None:
                    # Force-keep the test center atoms, randomly drop the rest
                    keep_mask = np.zeros(dpos_init_size, dtype=bool)
                    keep_mask[self.id_list_in_dpos] = True
                    need = target_ids_size - keep_mask.sum()
                    if need > 0:
                        candidates = np.flatnonzero(~keep_mask)
                        picked = self.rng.choice(candidates, size=need, replace=False)
                        keep_mask[picked] = True

                    # Apply mask to dpos and avoid_atoms 
                    self.dpos = self.dpos[keep_mask, ...]
                    avoid_atoms = avoid_atoms[keep_mask]

                    # Remap test atom indices to the new compact dpos
                    remap = -np.ones(dpos_init_size, dtype=int)
                    remap[np.flatnonzero(keep_mask)] = np.arange(keep_mask.sum())
                    self.id_list_in_dpos = remap[self.id_list_in_dpos]

                else: # case: train set with test_tol == 1
                    # no test centers to preserve: drop atoms at random.
                    picked_idx = self.rng.choice(dpos_init_size, size=target_ids_size, replace=False)
                    self.dpos = self.dpos[picked_idx, ...]
                    avoid_atoms = avoid_atoms[picked_idx]


                # Density (and thus neighborhood radius) changed: re-estimate it (minmax bounding box).
                hull_volume = (np.max(self.dpos[:, 0]) - np.min(self.dpos[:, 0])) * (np.max(self.dpos[:, 1]) - np.min(self.dpos[:, 1])) * (np.max(self.dpos[:, 2]) - np.min(self.dpos[:, 2]))
                overall_density = (len(self.dpos) / hull_volume) # average density
                r = ((self.num_points * 3) / (overall_density * np.pi * 4))**(1./3.) 
                self.max_dist_estimate = r * 1.2 # be more conservative
            else:
                print("Detection efficiency of 1 (100%)")

            if gaussian_noise !=0:

                if isotropic_noise:
                    noise = truncnorm.rvs(-2, 2, loc=0, scale=gaussian_noise, size=(self.dpos.shape[0],3), random_state=11) # truncated normal distribution limited to 2x std deviation
                else: # anisotropic: z-noise is set to 1/5 of the in-plane noise.
                    noise_xy = truncnorm.rvs(-2, 2, loc=0, scale=gaussian_noise, size=(self.dpos.shape[0],2), random_state=11) # truncated normal distribution limited to 2x std deviation
                    noise_z = truncnorm.rvs(-2, 2, loc=0, scale=gaussian_noise/5, size=(self.dpos.shape[0],1), random_state=22) # truncated normal distribution limited to 2x std deviation
                    noise = np.hstack((noise_xy, noise_z))
            
                self.dpos[:,0:3] += noise

            else:
                print("No noise added to atomic positions")

            if  gaussian_noise !=0 or det_efficiency != 1: # rebuild kdtree as underlying data changed
                self.kd_tree = KDTree(self.dpos[:, 0:3], leafsize=15)

                # Ensure test atoms are in the query set when dist_to_surface > 0
                if self.id_list_in_dpos is not None:
                    avoid_atoms[self.id_list_in_dpos] = False

                # Rebuild dpos_query and current mapping
                if self.dist_to_surface == 0:
                    self.dpos_query = self.dpos
                else:
                    self.dpos_query = self.dpos[~avoid_atoms]
                self.dpos_query_size = self.dpos_query.shape[0]

                # Recompute test neighborhoods & mask from the same atoms (post-sim coords)
                if self.test_tol < 1:
                    centers_post = self.dpos[self.id_list_in_dpos, 0:3]
                    dd, self.ii_test = self.kd_tree.query(centers_post, k=self.num_points, workers=4)
                    self.is_test_atom = np.zeros(self.dpos.shape[0], dtype=bool)
                    self.is_test_atom[self.ii_test.reshape(-1)] = True

                else:
                    if self.partition == "test": # recompute test set (post-sim coords)

                        self.rng_test = np.random.default_rng(seed=test_set_seed) 
                        self.id_list = self.rng_test.choice(self.dpos_query_size, self.test_size, replace=False)

                        center_atom = self.dpos_query[self.id_list, 0:3]
                        dd, self.ii_test = self.kd_tree.query(center_atom, k=self.num_points, workers=4)

        # build the pool of valid training centres (leakage-controlled with test_tol parameter)
        # A center is valid if its neighborhood/environment overlaps the union of test set atoms by at most ``allowed_overlap`` atoms. Only needed for the training partition under leakage control.
        if self.test_tol < 1 and self.partition == "train":
            chunk_s = 100000 

            valid_train_ids_max = self.train_size*epochs # target pool size

            rnd_id = self.rng.choice(self.dpos_query_size, self.dpos_query_size, replace=False) # shuffled query space ids
            chunks = int(ceil(self.dpos_query_size / chunk_s))

            self.valid_train_ids = [] # container for valid ids

            id_break = 0
            id_break_max = 50
            for c in range(chunks):
                sl = slice(c*chunk_s, min((c+1)*chunk_s, self.dpos_query_size))
                centers_idx = rnd_id[sl]                      # indices into dpos_query

                center_atom_check = self.dpos_query[centers_idx, 0:3]
                dd, ii = self.kd_tree.query(center_atom_check, k=self.num_points, workers=-1)

                counts = self.is_test_atom[ii].sum(axis=1)    
                valid_mask = counts <= self.allowed_overlap   

                self.valid_train_ids.extend(centers_idx[valid_mask].tolist())

                print(f"Valid ids progress {len(self.valid_train_ids)/valid_train_ids_max}")

                if len(self.valid_train_ids) >= valid_train_ids_max:
                    break
                if id_break == id_break_max:
                    print("Train ids break reached")
                    break
                id_break += 1

            print("Total valid train_ids " + str(len(self.valid_train_ids)) + ". (unique " + str(len(np.unique(self.valid_train_ids)))+  ") Ratio of valid ids compared to target: " + str(len(self.valid_train_ids) / valid_train_ids_max))

            self.valid_train_ids = np.array(self.valid_train_ids)

            unique_elements, counts = np.unique(self.dpos_query[self.valid_train_ids][:, -1], return_counts=True)
            counts_sum = sum(counts)
            phase_dict_ratio = {key: count / counts_sum for key, count in zip(unique_elements, counts)}
            print("Ratio of phases in valid ids: " + str(phase_dict_ratio))
        
        self.avoid_atoms = avoid_atoms

    def __getitem__(self, item): # item only used for test data

        if not hasattr(self, 'worker_rng_data'):
            self.worker_rng_data, self.worker_rng_cbc = get_worker_rngs(self.seed)
            self.worker_rng_aug = np.random.default_rng() 


        if self.partition == "train":
            
            if self.test_tol < 1:
                rnd_id = self.worker_rng_data.choice(self.valid_train_ids, 1, replace=True)[0]
            else:
                rnd_id = self.worker_rng_data.integers(0, self.dpos_query_size, 1)[0]
            
            center_atom = self.dpos_query[rnd_id, 0:3]
            dd, ii = self.kd_tree.query(center_atom, k=self.num_points, workers=1)
            ii_anchor = ii.copy()

            data = self.dpos[ii].copy()

        elif self.partition == "test":
            data = self.dpos[self.ii_test[item]].copy() # kd_tree was queried during init => constant data

        label = data[0,-1].copy()  # take label 

        pointcloud = data[:,0:-1].copy() # remove label indicator

        pointcloud[:,0:3] = normalize_unit_sphere_fixed(pointcloud[:,0:3], self.max_dist_estimate).astype("float32")

        # building the positive environment for the anchor. Only needed for self-supervised model
        if self.num_positives != 0: # return original and transformed dataset

            pointcloud_transformed = []
            for n in range(self.num_positives):

                if self.close_by_compare !=0  and self.partition == "train": 

                    max_idx =  max(2, int(self.num_points * self.close_by_compare) + 1)  # upper bound
                    cbc_candidates = ii_anchor[1:max_idx]                          
                    cbc_candidates = cbc_candidates[~self.avoid_atoms[cbc_candidates]] # remove potential surface near atoms so they are not sampled as centers

                    valid_cbc = False # valid_cbc are atoms that also pass the test_tol test
                    for id_cbc in self.worker_rng_cbc.permutation(cbc_candidates):
                        center_atom_cbc =  self.dpos[id_cbc, 0:3]
                        dd, ii_cbc = self.kd_tree.query(center_atom_cbc, k=self.num_points, workers=1)
                        if self.test_tol < 1:             # check if overlap ok
                            overlap = self.is_test_atom[ii_cbc].sum()
                            if overlap > self.allowed_overlap:
                                continue                 # reject & try next in list
                            else:
                                valid_cbc = True
                                break
                        else:
                            valid_cbc = True
                            break

                    if valid_cbc == False: # use anchor as fallback
                        ii_cbc = ii_anchor.copy()
                        id_cbc = ii_anchor[0]

                    data = self.dpos[ii_cbc].copy()
                    pointcloud_cbc = data[:,0:-1] # remove label indicator

                    pointcloud_cbc[:,0:3] = normalize_unit_sphere_fixed(pointcloud_cbc[:,0:3], self.max_dist_estimate).astype("float32")

                    pointcloud_transformed.append(transform_data_single(pointcloud_cbc, rotate=self.rotation, jitter=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)) 
                else:
                    pointcloud_transformed.append(transform_data_single(pointcloud, rotate=self.rotation,  jitter=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)) 

            if self.augmentation and self.partition == "train": 
                pointcloud = rotate_pointcloud(pointcloud, mode=self.rotation, rng=self.worker_rng_aug)
                if self.jitter != 0:
                    pointcloud = jitter_pointcloud(pointcloud, sigma=self.jitter, radius=self.max_dist_estimate, rng=self.worker_rng_aug)

            return pointcloud, pointcloud_transformed, label

        else: # supervised case
            if self.augmentation:
                if self.partition == "train":  # supervised case
                    pointcloud = rotate_pointcloud(pointcloud, mode=self.rotation,rng=self.worker_rng_aug)  
                    if self.jitter != 0:
                        pointcloud = jitter_pointcloud(pointcloud, sigma=self.jitter, radius=self.max_dist_estimate,rng=self.worker_rng_aug)  
                elif self.partition == "test":
                    raise ValueError("Augmentation of test set not implemented")

            return pointcloud, label


    def __len__(self):

        if self.partition == "train":
            return self.train_size
        elif self.partition == "test":
            return self.test_size



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

        dataset_out = APT_basic(
            data_path=config.data_path,
            num_points=config.num_points,
            input_features=config.input_features,
            elements=config.elements,
            seed=config.seed,
            train_size=config.train_size,
            test_size=config.test_size,
            num_positives=num_positives,
            test_tol=config.test_tol, 
            partition=partition,
            phase_dict=config.phase_dict,
            close_by_compare=close_by_compare,
            augmentation=augmentation,
            rotation=config.rotation,
            jitter=config.jitter,
        )

    elif config.dataset_mode == "APT_dynamic":
        dataset_out = APT_dynamic(
            data_path=config.data_path,
            log_path=config.log_path,
            num_points=config.num_points,
            input_features=config.input_features,
            elements=config.elements,
            phase_identifier=config.phase_identifier,
            phase_dict=config.phase_dict,
            seed=config.seed,
            train_size=config.train_size,
            test_size=config.test_size,
            num_positives=num_positives, 
            partition=partition,
            dist_to_surface=config.dist_to_surface,
            close_by_compare=close_by_compare,
            test_tol=config.test_tol,
            augmentation=augmentation,
            sim_data=config.sim_data,
            gaussian_noise=config.gaussian_noise,
            det_efficiency=config.det_efficiency,
            isotropic_noise=config.isotropic_noise,
            epochs=config.epochs
            )

    return dataset_out


