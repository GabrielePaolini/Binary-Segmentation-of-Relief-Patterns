# function to turn points loaded via pdal into a pyg Data object, with additional channels
import numpy as np
from torch_geometric.data import Data


def lidar_hd_pre_transform(points):
    """Turn pdal points into torch-geometric Data object.

    Builds a composite (average) color channel on the fly.     
    Calculate NDVI on the fly.

    Args:
        las_filepath (str): path to the LAS file.

    Returns:
        Data: the point cloud formatted for later deep learning training.

    """
    # Positions and base features
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    normals = np.asarray([points["NormalsX"], points["NormalsY"], points["NormalsZ"]], dtype=np.float32).transpose()
    #dummy_values = np.ones_like(points["Local_Depth"], dtype=np.float32)
    Local_Depth_Squared = points["Local_Depth"]**2

    x = np.stack(
        [
            #points["Azimuth"],
            #points["Elevation"],
            #dummy_values
            points["Local_Depth"],
            Local_Depth_Squared,
            #points["Curvature"],
        ], axis=0
    ).transpose()
    #x_features_names = ["Curvature",]
    #x_features_names = ["Azimuth", "Elevation", "Local_Depth"]
    x_features_names = ["Local_Depth", "Local_Depth_Squared"]
    y = points["Classification"]
    normals = normals

    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names, normals=normals)

    return data
