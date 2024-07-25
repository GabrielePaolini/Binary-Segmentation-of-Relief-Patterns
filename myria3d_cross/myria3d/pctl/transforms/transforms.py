import math
import re
from typing import Dict, List, Tuple, Union
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, LinearTransformation

from myria3d.utils import utils
import laspy

log = utils.get_logger(__name__)

COMMON_CODE_FOR_ALL_ARTEFACTS = 65


class ToTensor(BaseTransform):
    """Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys: List[str] = ["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


def subsample_data(data, num_nodes, choice):
    # TODO: get num_nodes from data.num_nodes instead to simplify signature
    for key, item in data:
        if key == "num_nodes":
            data.num_nodes = choice.size(0)
        elif bool(re.search("edge", key)):
            continue
        elif torch.is_tensor(item) and item.size(0) == num_nodes and item.size(0) != 1:
            data[key] = item[choice]
    return data


class MaximumNumNodes(BaseTransform):
    def __init__(self, num: int):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes <= self.num:
            return data

        choice = torch.randperm(data.num_nodes)[: self.num]
        data = subsample_data(data, num_nodes, choice)

        return data


class MinimumNumNodes(BaseTransform):
    def __init__(self, num: int):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes >= self.num:
            return data

        choice = torch.cat(
            [torch.randperm(num_nodes) for _ in range(math.ceil(self.num / num_nodes))],
            dim=0,
        )[: self.num]

        data = subsample_data(data, num_nodes, choice)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num}"


class CopyFullPos:
    """Make a copy of the original positions - to be used for test and inference."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["pos_copy"] = data["pos"].clone()
        return data


class CopyFullPreparedTargets:
    """Make a copy of all, prepared targets - to be used for test."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["transformed_y_copy"] = data["y"].clone()
        return data


class CopySampledPos(BaseTransform):
    """Make a copy of the unormalized positions of subsampled points - to be used for test and inference."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["pos_sampled_copy"] = data["pos"].clone()
        return data


class StandardizeRGBAndIntensity(BaseTransform):
    """Standardize RGB and log(Intensity) features."""

    def __call__(self, data: Data):
        idx = data.x_features_names.index("Intensity")
        # Log transform to be less sensitive to large outliers - info is in lower values
        data.x[:, idx] = torch.log(data.x[:, idx] + 1)
        data.x[:, idx] = self.standardize_channel(data.x[:, idx])
        idx = data.x_features_names.index("rgb_avg")
        data.x[:, idx] = self.standardize_channel(data.x[:, idx])
        return data

    def standardize_channel(self, channel_data: torch.Tensor, clamp_sigma: int = 3):
        """Sample-wise standardization y* = (y-y_mean)/y_std. clamping to ignore large values."""
        mean = channel_data.mean()
        std = channel_data.std() + 10**-6
        if torch.isnan(std):
            std = 1.0
        standard = (channel_data - mean) / std
        clamp = clamp_sigma * std
        clamped = torch.clamp(input=standard, min=-clamp, max=clamp)
        return clamped


class NullifyLowestZ(BaseTransform):
    """Center on x and y axis only. Set lowest z to 0."""

    def __call__(self, data):
        data.pos[:, 2] = data.pos[:, 2] - data.pos[:, 2].min()
        return data


class NormalizePosSubTile(BaseTransform):
    """
    Normalizes xy in [-1;1] range by scaling the whole point cloud (including z dim).
    XY are expected to be centered on zéro.

    """

    def __init__(self, subtile_width=50):
        half_subtile_width = subtile_width / 2
        self.scaling_factor = 1 / half_subtile_width

    def __call__(self, data):
        data.pos = data.pos * self.scaling_factor
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
    

class NormalizePos(BaseTransform):
    """
    Normalizes xy in [-1;1] range by scaling the whole point cloud (including z dim).
    XY are expected to be centered on zéro.

    """
    def __call__(self, data):
        max_abs_value = data.pos.abs().max()
        scaling_factor = 1 / max_abs_value  # if max_abs_value != 0 else 1
        data.pos = data.pos * scaling_factor
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
    

class AddGaussianNoise(BaseTransform):
    def __call__(self, data):
        mu, sigma = 0,0.2
        data.pos = data.pos + np.random.default_rng().normal(mu, sigma, data.pos.shape)
        return data


class AlignToAxis(BaseTransform):
    """
    Allinea la point cloud ad un asse di riferimento globale. 
    """
    def __init__(self):
        self.target_axis = torch.tensor([0,0,1], dtype=torch.float32)
        self.idx = 0

    def __call__(self, data):
        
        """
        filename = f"./patch_{self.idx}.las"
        pointrecord = laspy.create(file_version="1.4", point_format=3)
        vertices = data.pos.numpy()
        pointrecord.header.offsets = np.min(vertices, axis=0)
        point_count = vertices.shape[0]
        pointrecord.header.point_count = point_count
        pointrecord.x = vertices[:, 0]
        pointrecord.y = vertices[:, 1]
        pointrecord.z = vertices[:, 2]
        pointrecord.add_extra_dim(laspy.ExtraBytesParams(name="Curvature", type=np.float64))
        pointrecord["Curvature"] = data.x.numpy().squeeze()
        pointrecord.write(filename)
        self.idx += 1
        """

        # Align to Z axis
        cov_matrix = torch.matmul(data.pos.T, data.pos) / data.pos.size(0)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)   # Normalizzati
        principal_axis = eigenvectors[:, torch.argmin(eigenvalues)]
        if torch.dot(principal_axis, self.target_axis) < 0:
            principal_axis *= -1
        v = torch.cross(principal_axis, self.target_axis)
        c = torch.dot(principal_axis, self.target_axis)
        s = torch.linalg.norm(v)
        kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float32)
        rotation_matrix = torch.eye(3) + kmat + torch.matmul(kmat,kmat) * ((1 -c) / (s ** 2))
        data.pos = torch.matmul(data.pos, rotation_matrix)

        # Check Z axis to see if it is aligned
        cov_matrix = torch.matmul(data.pos.T, data.pos) / data.pos.size(0)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        principal_axis = eigenvectors[:, torch.argmin(eigenvalues)]
        if torch.dot(principal_axis, self.target_axis) < 0:
            principal_axis *= -1
        #print(f"Is Z axis aligned? {torch.rad2deg(torch.acos(torch.dot(principal_axis, torch.tensor([0,0,1], dtype=torch.float))))}")

        # Compute inertia tensor for the XY projection
        I_xx = torch.sum(data.pos[:,1]**2)
        I_yy = torch.sum(data.pos[:,0]**2)
        I_xy = -torch.sum(data.pos[:,0] * data.pos[:,1])
        inertia_tensor = torch.tensor([[I_xx, I_xy], [I_xy, I_yy]])
        # Secondary principal axis
        eigenvalues, eigenvectors = torch.linalg.eigh(inertia_tensor)
        secondary_axis = eigenvectors[:, torch.argmax(eigenvalues)]
        # Determine the angle to rotate the secondary principal axis to align with the X-axis
        angle_to_x_axis = torch.atan2(secondary_axis[1], secondary_axis[0])
        # Construct the rotation matrix around the Z-axis
        cos_angle = torch.cos(-angle_to_x_axis)
        sin_angle = torch.sin(-angle_to_x_axis)
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            self.target_axis
        ])
        # Apply the rotation to the point cloud
        data.pos = torch.matmul(data.pos, rotation_matrix)   
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

  
class SO3Augmentation(BaseTransform):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval (functional name: :obj:`random_rotate`).

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """
    def __init__(
        self,
        degrees: Union[Tuple[float, float], float],
        axis: int = 0,
    ) -> None:
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        assert data.normals is not None

        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(-1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        data.normals = torch.matmul(data.normals, torch.tensor(matrix))
        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.degrees}, '
                f'axis={self.axis})')


class TargetTransform(BaseTransform):
    """
    Make target vector based on input classification dictionnary.

    Example:
    Source : y = [6,6,17,9,1]
    Pre-processed:
    - classification_preprocessing_dict = {17:1, 9:1}
    - y' = [6,6,1,1,1]
    Mapped to consecutive integers:
    - classification_dict = {1:"unclassified", 6:"building"}
    - y'' = [1,1,0,0,0]

    """

    def __init__(
        self,
        classification_preprocessing_dict: Dict[int, int],
        classification_dict: Dict[int, str],
    ):
        self._set_preprocessing_mapper(classification_preprocessing_dict)
        self._set_mapper(classification_dict)

        # Set to attribute to log potential type errors
        self.classification_dict = classification_dict
        self.classification_preprocessing_dict = classification_preprocessing_dict

    def __call__(self, data: Data):
        data.y = self.transform(data.y)
        return data

    def transform(self, y):
        y = self.preprocessing_mapper(y)
        try:
            y = self.mapper(y)
        except TypeError as e:
            log.error(
                "A TypeError occured when mapping target from arbitrary integers "
                "to consecutive integers (0-(n-1)) using the provided classification_dict "
                "This usually happens when an unknown classification code was encounterd. "
                "Check that all classification codes in your data are either "
                "specified via the classification_dict "
                "or transformed into a specified code via the preprocessing_mapper. \n"
                f"Current classification_dict: \n{self.classification_dict}\n"
                f"Current preprocessing_mapper: \n{self.classification_preprocessing_dict}\n"
                f"Current unique values in preprocessed target array: \n{np.unique(y)}\n"
            )
            raise e
        return torch.LongTensor(y)

    def _set_preprocessing_mapper(self, classification_preprocessing_dict):
        """Set mapper from source classification code to another code."""
        d = {key: value for key, value in classification_preprocessing_dict.items()}
        self.preprocessing_mapper = np.vectorize(lambda class_code: d.get(class_code, class_code))

    def _set_mapper(self, classification_dict):
        """Set mapper from source classification code to consecutive integers."""
        d = {
            class_code: class_index
            for class_index, class_code in enumerate(classification_dict.keys())
        }
        # Here we update the dict so that code 65 remains unchanged.
        # Indeed, 65 is reserved for noise/artefacts points, that will be deleted by transform "DropPointsByClass".
        d.update({65: 65})
        self.mapper = np.vectorize(lambda class_code: d.get(class_code))


class DropPointsByClass(BaseTransform):
    """Drop points with class -1 (i.e. artefacts that would have been mapped to code -1)"""

    def __call__(self, data):
        points_to_drop = torch.isin(data.y, COMMON_CODE_FOR_ALL_ARTEFACTS)
        if points_to_drop.sum() > 0:
            points_to_keep = torch.logical_not(points_to_drop)
            data = subsample_data(data, num_nodes=data.num_nodes, choice=points_to_keep)
            # Here we also subsample these idx since we do not need to interpolate these points back
            if "idx_in_original_cloud" in data:
                data.idx_in_original_cloud = data.idx_in_original_cloud[points_to_keep]
        return data
