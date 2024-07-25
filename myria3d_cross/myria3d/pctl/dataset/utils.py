import glob
import json
from pathlib import Path
import subprocess as sp
from numbers import Number
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import pdal
from scipy.spatial import cKDTree
import potpourri3d as pp3d
from collections import Counter

SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
DATA_PATHS_BY_SPLIT_DICT_TYPE = Dict[SPLIT_TYPE, List[str]]

def find_file_in_dir(data_dir: str, basename: str) -> str:
    """Query files matching a basename in input_data_dir and its subdirectories.
    Args:
        input_data_dir (str): data directory
    Returns:
        [str]: first file path matching the query.
    """
    query = f"{data_dir}/**/{basename}"
    files = glob.glob(query, recursive=True)
    return files[0]

def get_mosaic_of_centers(tile_width: Number, subtile_width: Number, subtile_overlap: Number = 0):
    if subtile_overlap < 0:
        raise ValueError("datamodule.subtile_overlap must be positive.")

    xy_range = np.arange(
        subtile_width / 2,
        tile_width + (subtile_width / 2) - subtile_overlap,
        step=subtile_width - subtile_overlap,
    )
    return [np.array([x, y]) for x in xy_range for y in xy_range]

def pdal_read_las_array(las_path: str, epsg: str):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path
        epsg (str): epsg to force the reading with

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones (not true), with dict-like access.

    """
    p1 = pdal.Pipeline() | get_pdal_reader(las_path, epsg)
    p1.execute()
    return p1.arrays[0]

def pdal_read_las_array_as_float32(las_path: str, epsg: str):
    """Read LAS as a a named array, casted to floats."""
    arr = pdal_read_las_array(las_path, epsg)
    #formats = ["f4" if name != "TileIdx" else "i4" for name in arr.dtype.names]
    formats = [arr.dtype[name] for name in arr.dtype.names]
    all_floats = np.dtype({"names": arr.dtype.names, "formats": formats})
    return arr.astype(all_floats)

def get_metadata(las_path: str) -> dict:
    """ returns metadata contained in a las file
    Args:
        las_path (str): input LAS path to get metadata from.
    Returns:
        dict : the metadata.
    """
    pipeline = pdal.Reader.las(filename=las_path).pipeline()
    pipeline.execute()
    return pipeline.metadata

# Extra dims hard-coded! Forse aggiungendole al config file si rende il codice più leggibile e versatile
def get_pdal_reader(las_path: str, epsg: str) -> pdal.Reader.las:
    """Standard Reader.
    Args:
        las_path (str): input LAS path to read.
        epsg (str): epsg to force the reading with
    Returns:
        pdal.Reader.las: reader to use in a pipeline.

    """

    if epsg :
        # if an epsg in provided, force pdal to read the lidar file with it
        try :  # epsg can be added as a number like "2154" or as a string like "EPSG:2154"
            int(epsg)
            return pdal.Reader.las(
                filename=las_path,
                nosrs=True,
                override_srs=f"EPSG:{epsg}",
                #extra_dims="Azimuth=float64,Elevation=float64,Local_Depth=float64,TileIdx=float64"
            )
        except ValueError:
            return pdal.Reader.las(
                filename=las_path,
                nosrs=True,
                override_srs=epsg,
                #extra_dims="Azimuth=float64,Elevation=float64,Local_Depth=float64,TileIdx=float64"
            )

    try :
        if get_metadata(las_path)['metadata']['readers.las']['srs']['compoundwkt']:
            # read the lidar file with pdal default
            return pdal.Reader.las(filename=las_path)
    except Exception:
        pass  # we will go to the "raise exception" anyway

    raise Exception("No EPSG provided, neither in the lidar file or as parameter")

def get_pdal_info_metadata(las_path: str) -> Dict:
    """Read las metadata using pdal info
    Args:
        las_path (str): input LAS path to read.
    Returns:
        (dict): dictionary containing metadata from the las file
    """
    r = sp.run(["pdal", "info", "--metadata", las_path], capture_output=True)
    if r.returncode == 1:
        msg = r.stderr.decode()
        raise RuntimeError(msg)

    output = r.stdout.decode()
    json_info = json.loads(output)

    return json_info["metadata"]

def read_npy(file_path, return_face = True, **kwargs):
    # Load the npy file (assuming it contains a dictionary)
    data_dict = np.load(file_path, allow_pickle=True).item()
    # Extract face array
    #try:
    #    faces = data_dict["Faces"] if return_face else None
    #    del data_dict["Faces"]
    #except KeyError as e:
    #    print("Faces record not present in numpy array {}".format(e))
    # Determine dtype programmatically
    dtype = [(key, value.dtype) for key,value in data_dict.items()]
    # Create an empty structured array with the determined dtyp
    array_length = len(next(iter(data_dict.values())))
    structured_array = np.zeros(array_length, dtype=dtype)
    # Fill the structured array with data from the dictionary
    for key in data_dict:
        structured_array[key] = data_dict[key]
    return structured_array #, faces

# hdf5, iterable
file_readers = {
    "las": pdal_read_las_array_as_float32, 
    "npy": read_npy,
}

def split_cloud_into_samples(
    data_path: str,
    tile_width: Number,
    subtile_width: Number,
    epsg: str,
    subtile_overlap: Number = 0,
    **kwargs,
):
    """Split LAS/NPY point cloud into samples.

    Args:
        las_path (str): path to raw LAS file
        tile_width (Number): width of input LAS file
        subtile_width (Number): width of receptive field.
        epsg (str): epsg to force the reading with
        subtile_overlap (Number, optional): overlap between adjacent tiles. Defaults to 0.

    Yields:
        _type_: idx_in_original_cloud, and points of sample in pdal input format casted as floats.

    """
    points = pdal_read_las_array_as_float32(data_path, epsg)
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    kd_tree = cKDTree(pos[:, :2] - pos[:, :2].min(axis=0))
    XYs = get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap=subtile_overlap)
    for center in XYs:
        radius = subtile_width // 2  # Square receptive field.
        minkowski_p = np.inf
        sample_idx = np.array(kd_tree.query_ball_point(center, r=radius, p=minkowski_p))
        if not len(sample_idx):
            # no points in this receptive fields
            continue
        
        sample_points = points[sample_idx]  # Dal file las ottengo posizione e features (non curvatura, ma rgb etc)
        yield sample_idx, sample_points

def pre_filter_below_n_points(data, min_num_nodes=1):
    return data.pos.shape[0] < min_num_nodes

def get_las_paths_by_split_dict(
    data_dir: str, split_csv_path: str
) -> DATA_PATHS_BY_SPLIT_DICT_TYPE:
    las_paths_by_split_dict: DATA_PATHS_BY_SPLIT_DICT_TYPE = {}
    split_df = pd.read_csv(split_csv_path)
    for phase in ["train", "val", "test"]:
        basenames = split_df[split_df.split == phase].basename.tolist()
        # Reminder: an explicit data structure with ./val, ./train, ./test subfolder is required.
        las_paths_by_split_dict[phase] = [str(Path(data_dir) / phase / b) for b in basenames]

    if not las_paths_by_split_dict:
        raise FileNotFoundError(
            (
                f"No basename found while parsing directory {data_dir}"
                f"using {split_csv_path} as split CSV."
            )
        )

    return las_paths_by_split_dict

def split_cloud_into_GVD(
    data_path: str,
    epsg: str,
    *args,
    **kwargs
):
    """Split LAS/NPY point cloud into samples.

    Args:
        data_path (str): path to raw LAS file

    Yields:
        _type_: idx_in_original_cloud, and points of sample in pdal input format casted as floats.

    """
    points = pdal_read_las_array_as_float32(data_path, epsg)

    # Preprocessing: allineamento tutta la pc
    #pos = np.asarray([points.x, points.y, points.z], dtype=np.float32).transpose()
    #aligned_pos = align_point_cloud(pos)
    #points.x = aligned_pos[:,0]
    #points.y = aligned_pos[:,1]
    #points.z = aligned_pos[:,2]

    tile_idx = points["TileIdx"]
    num_tiles = np.unique(tile_idx)

    for tile in num_tiles:
        sample_idx = np.where(tile_idx == tile)[0]
        if not len(sample_idx):
            continue
        
        sample_points = points[sample_idx]  # Dal file las ottengo posizione e features (non curvatura, ma rgb etc)
        yield sample_idx, sample_points

def sample_whole_cloud(
    data_path: str,
    epsg: str,
    *args,
    **kwargs
):
    points = pdal_read_las_array_as_float32(data_path, epsg)
    nb_points = points.x.shape[0]
    sample_idx = np.arange(nb_points)
    sample_points = points[sample_idx]
    yield sample_idx, sample_points

def voronoi_refine(points: np.ndarray, n_centroids: Number = 3, minimum_pts: Number = 300):
    vertices = np.asarray(points["pos"])
    solver = pp3d.PointCloudHeatSolver(vertices)
    start_idx = np.random.randint(1, vertices.shape[0] + 1, 1)

    # Find centroids
    fps = farthest_point_sampling(vertices, solver, n_sample=n_centroids, start_idx=start_idx[0])
    
    # Extend scalar values to compute Voronoi diagram
    tile_idx = solver.extend_scalar(fps, list(range(fps.shape[0])))
    
    # Smoothing borders
    kdtree = cKDTree(vertices)
    for idx,point in enumerate(vertices):
        sample_idx = np.array(kdtree.query(point, k=32)[1])
        rounded_labels = np.rint(tile_idx[sample_idx])
        counter = Counter(rounded_labels)
        smoothed_label = max(counter, key=counter.get)
        tile_idx[idx] = smoothed_label
    
    num_tiles = np.unique(tile_idx)
    for i,tile in enumerate(num_tiles):
        sample_idx = np.where(tile_idx == tile)[0]
        if not len(sample_idx):
            continue
        elif len(sample_idx) <= minimum_pts:
            tile_idx[sample_idx] = num_tiles[i-1]
            continue
    
        yield sample_idx

def farthest_point_sampling(arr, solver, n_sample, start_idx=None):
    """Farthest Point Sampling without the need to compute all pairs of distance.

    Parameters
    ----------
    arr : numpy array
        The positional array of shape (n_points, n_dim)
    n_sample : int
        The number of points to sample.
    start_idx : int, optional
        If given, appoint the index of the starting point,
        otherwise randomly select a point as the start point.
        (default: None)

    Returns
    -------
    numpy array of shape (n_sample,)
        The sampled indices.
    """
    n_points, n_dim = arr.shape

    if (start_idx is None) or (start_idx < 0):
        start_idx = np.random.randint(0, n_points)

    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)
    
    for _ in range(n_sample - 1):
        dist_to_current_point = solver.compute_distance(sampled_indices[-1])
        #current_point = arr[sampled_indices[-1]]
        #dist_to_current_point = np.linalg.norm(arr - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)

    return np.array(sampled_indices)

def align_point_cloud(points):
    target_axis = np.array([0,0,1], dtype=np.float32)
    # Align to Z axis
    cov_matrix = np.matmul(np.transpose(points), points) / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)   # Normalizzati
    principal_axis = eigenvectors[:, np.argmin(eigenvalues)]
    if np.dot(principal_axis, target_axis) < 0:
        principal_axis *= -1
    v = np.cross(principal_axis, target_axis)
    c = np.dot(principal_axis, target_axis)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)
    rotation_matrix = np.eye(3) + kmat + np.matmul(kmat,kmat) * ((1 -c) / (s ** 2))
    points = np.matmul(points, rotation_matrix)

    # Check Z axis to see if it is aligned
    cov_matrix = np.matmul(np.transpose(points), points) / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    principal_axis = eigenvectors[:, np.argmin(eigenvalues)]
    if np.dot(principal_axis, target_axis) < 0:
        principal_axis *= -1
    #print(f"Is Z axis aligned? {torch.rad2deg(torch.acos(torch.dot(principal_axis, torch.tensor([0,0,1], dtype=torch.float))))}")

    # Compute inertia tensor for the XY projection
    I_xx = np.sum(points[:,1]**2)
    I_yy = np.sum(points[:,0]**2)
    I_xy = -np.sum(points[:,0] * points[:,1])
    inertia_tensor = np.array([[I_xx, I_xy], [I_xy, I_yy]])
    # Secondary principal axis
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    secondary_axis = eigenvectors[:, np.argmax(eigenvalues)]
    # Determine the angle to rotate the secondary principal axis to align with the X-axis
    angle_to_x_axis = np.arctan2(secondary_axis[1], secondary_axis[0])
    # Construct the rotation matrix around the Z-axis
    cos_angle = np.cos(-angle_to_x_axis)
    sin_angle = np.sin(-angle_to_x_axis)
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        target_axis
    ])
    # Apply the rotation to the point cloud
    points = np.matmul(points, rotation_matrix)   
    return points
