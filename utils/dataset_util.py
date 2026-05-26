
import numpy as np
import pandas as pd
from utils.apt_import import get_dpos
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation as R

from scipy.stats import truncnorm
import torch

"""
HELPER FUNCTIONS
"""

def get_worker_rngs(base_seed):

    worker_info = torch.utils.data.get_worker_info()
    
    if worker_info is not None:
        seed_to_use = worker_info.seed
    else:
        seed_to_use = base_seed         # Fallback for num_workers=0

    ss = np.random.SeedSequence(seed_to_use)
    child_seeds = ss.spawn(2)

    rng_1 = np.random.default_rng(child_seeds[0])
    rng_2 = np.random.default_rng(child_seeds[1])
    
    return rng_1, rng_2


def load_raw_data(data_path, elements, dissolve_n=True):

    extension = data_path.split(".")[-1].lower()

    if extension == "csv":
        dpos = pd.read_csv(data_path)
        dpos = dpos.reset_index(drop=True)

        if dissolve_n: # Further splits e.g. N:2 into N:1 and N:1 (identical rows apart from id)
            dpos = dpos.loc[dpos.index.repeat(dpos['n'])].copy() 
            dpos['n'] = 1
            dpos["id"] =  np.arange(dpos.x.size)

    elif extension == "epos":
        epos_path = data_path
        range_path = data_path.split(".")[0] + ".rrange"
        dpos = get_dpos(epos_path, range_path, dissolve_n=dissolve_n)

        if "label" not in dpos.columns: # dummy for epos data
            dpos["label"] = 0

    if elements != None:
        dpos = dpos[dpos.element.isin(elements)]  # restrict to selected elements

    return dpos


def get_data_raw(
    data_path,
    elements,
    phase_dict={"dummy":0},
    input_features = ["x", "y", "z", "element"],
    k=16,
    dissolve_n=True
):
    from scipy.spatial import ConvexHull

    dpos = load_raw_data(data_path, elements, dissolve_n)  # get raw data

    if elements == None:
        elements_features = pd.unique(dpos["element"])
    else:
        elements_features = elements

    print("Dpos size: " + str(dpos.x.size), flush=True)

    print(f"Phase dict is {phase_dict}")            

    dpos_label = dpos.label.replace(phase_dict).copy()  # replace strings with numbers for elements

    dpos = get_additional_features(dpos,input_features,elements_features) #modifies dpos to have the input_features. element will be one hot encoded
    dpos = pd.concat([dpos, dpos_label], axis=1) # add label info, will be removed later

    print("Dpos columns:" +str(dpos.columns))
    print("Content of labels")
    print(pd.value_counts(dpos["label"])/dpos.shape[0])

    dpos = dpos.to_numpy().astype("float32")
    print("size of dpos " + str(dpos.shape))

    hull_volume = ConvexHull(dpos[:, 0:3]).volume # rough approximation
    overall_density = (len(dpos) / hull_volume)
    r = ((k * 3) / (overall_density * np.pi * 4))**(1./3.) 
    max_dist_estimate = r * 1.2 # add 20% to be more conservative
    print("Using exp max dist estimate")

    # For rectangular data    
    # hull_volume = (np.max(dpos[:, 0]) - np.min(dpos[:, 0])) * (np.max(dpos[:, 1]) - np.min(dpos[:, 1])) * (np.max(dpos[:, 2]) - np.min(dpos[:, 2]))
    # overall_density = (len(dpos) / hull_volume) 
    # r = ((k * 3) / (overall_density * np.pi * 4))**(1./3.) # radius of sphere corresponding to aprox density & num_points
    # max_dist_estimate = r * 1.2  # add 20% to be more conservative
    # print("Using sim max dist estimate with " +str(max_dist_estimate))

    return dpos, max_dist_estimate



def get_additional_features(dpos, input_features, elements=None): #get dpos and return dataframe with needed values
    from sklearn.preprocessing import OneHotEncoder

    dpos_out = dpos[["x", "y", "z"]] #base features

    if "element" in input_features: # Adds element identification as feature, atoms of selected elements are added regardless

        element_one_hot = pd.get_dummies(dpos["element"])
        element_one_hot = element_one_hot[elements].astype("float32") # order columns based on user input
            
        dpos_out = pd.concat([dpos_out, element_one_hot], axis=1)

    if "mass_to_charge" in input_features: # Alternative to element

        Da = dpos[["Da"]]
        Da = (Da - Da.min()) / (Da.max() - Da.min())

        dpos_out = pd.concat([dpos_out, Da], axis=1)

    if "n" in input_features: 
        dpos_out["n"] = dpos["n"] / dpos["n"].max()


    print("Input features: " + str(list(dpos_out.columns.values))) 
    print("input dims: " + str(len(dpos_out.columns.values)))

    return dpos_out


def normalize_unit_sphere_fixed(points, radius):  # data (nbr_points, 3)
    centroid = np.mean(points, axis=0)
    points -= centroid
    points /= radius

    return points

def normalize_unit_sphere_fixed_batched(points, radius):
    centroid = np.mean(points, axis=1, keepdims=True)
    points -= centroid
    points /= radius

    return points


# Approximation of surface points/atoms. No guarantee of success
def get_surface_points_alpha_shape(dpos, dpos_query = None, initial_alpha=2, mesh_until_watertight = True, dist_to_surface=1.5,return_bool=False): 
    from scipy.spatial import KDTree
    import open3d as o3d
    import numpy as np

    alpha = initial_alpha
    pointcloud_size = dpos.shape[0]
    max_point_cloud_size = 4000000 
    if pointcloud_size > max_point_cloud_size:
        rng = np.random.default_rng()
        data_np_mesh = rng.choice(dpos[:,0:3], size=max_point_cloud_size, replace=False)
    else:
        data_np_mesh = dpos[:,0:3]

    pcd_ions = o3d.geometry.PointCloud()
    pcd_ions.points = o3d.utility.Vector3dVector(data_np_mesh)

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_ions) # Allows reuse of tetra_mesh if multiple alpha's are tested
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ions, alpha, tetra_mesh, pt_map)

    max_alpha = 5

    if mesh_until_watertight:
        while not mesh.is_watertight() and alpha < max_alpha:
            alpha += 0.1
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ions, alpha, tetra_mesh, pt_map)
            print(alpha, flush=True)
        if alpha >= max_alpha:
            print(f"Warning: Reached max_alpha ({max_alpha}) without achieving watertight mesh.")


    print("Final alpha: " + str(alpha))
    print("Mesh watertight: " + str(mesh.is_watertight()))
    
    pcd_surface = mesh.sample_points_uniformly(number_of_points=1000000)

    tree = KDTree(pcd_surface.points)

    if dpos_query is None:
        dist, _ = tree.query(dpos[:,0:3], k=1,distance_upper_bound=dist_to_surface, workers=-1)
    else:
        dist, _ = tree.query(dpos_query[:,0:3], k=1,distance_upper_bound=dist_to_surface,workers=-1)

    if return_bool:
        return (dist <= dist_to_surface)
    else:
        return (dist <= dist_to_surface).astype(int)

"""
DATA AUGMENTATION
"""

def jitter_pointcloud(pointcloud, sigma=0.01, clip=2, radius=1, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    coords = pointcloud[:, 0:3]
    
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    
    N_unique = unique_coords.shape[0]
    C = coords.shape[1]
    
    noise_unique = truncnorm.rvs(-clip, clip, loc=0, scale=sigma, size=(N_unique, C), random_state=rng) / radius
    
    noise_full = noise_unique[inverse_indices]
    
    pointcloud[:, 0:3] += noise_full
    
    return pointcloud

def rotate_pointcloud(pointcloud, mode="so3", rng=None):

    if rng is None:
        rng = np.random.default_rng()

    if mode == "none":
        return pointcloud
    elif mode == "so3":
        rotation_matrix = special_ortho_group.rvs(3, random_state=rng)
    elif mode == "z":
        d = rng.uniform(0, 360)
        rotation_matrix = R.from_euler('z', d, degrees=True).as_matrix()        

    pointcloud[:, 0:3] = pointcloud[:, 0:3].dot(rotation_matrix)  
    return pointcloud

# augmentation for positive environments
def transform_data_single(data, rotate="so3", jitter=0, radius=1, rng=None):

    data_t = data.copy()

    if rotate != None:
        data_t = rotate_pointcloud(data_t, mode=rotate, rng=rng)
    
    if jitter != 0:
        data_t = jitter_pointcloud(data_t, sigma=jitter, clip=2,radius=radius,rng=rng)

    return data_t
