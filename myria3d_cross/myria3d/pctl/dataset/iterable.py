from numbers import Number
from typing import Callable, Optional

import torch
from numpy.typing import ArrayLike
from torch.utils.data.dataset import IterableDataset
from torch_geometric.data import Data

from myria3d.pctl.dataset.utils import (
    pre_filter_below_n_points,
    split_cloud_into_samples,
    split_cloud_into_GVD,
    sample_whole_cloud,
    voronoi_refine,
)

SAMPLER = {"tiling": split_cloud_into_samples, "voronoi": split_cloud_into_GVD, "whole": sample_whole_cloud} # Dizionario key: nome metodo generazione "tile", value: funzione da utils per la generazione

from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform

class InferenceDataset(IterableDataset):
    """Iterable dataset to load samples from a single las file."""

    def __init__(
        self,
        las_file: str,
        epsg: str,
        sampling_method: str = "voronoi",
        points_pre_transform: Callable[[ArrayLike], Data] = lidar_hd_pre_transform,
        pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
        transform: Optional[Callable[[Data], Data]] = None,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        file_format: str = "las",
    ):
        self.las_file = las_file
        self.epsg = epsg
        self.sampling_method = sampling_method

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter
        self.transform = transform

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap = subtile_overlap

        self.file_format = file_format

    def __iter__(self):
        # Chiamata quando viene ritornata una batch dal dataloader 
        return self.get_iterator()

    def get_iterator(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""
        for idx_in_original_cloud, sample_points in SAMPLER[self.sampling_method](   # split_cloud_into_samples suddivide la point cloud in subtiles
            data_path=self.las_file,
            epsg=self.epsg,
            tile_width=self.tile_width,
            subtile_width=self.subtile_width,
            subtile_overlap=self.subtile_overlap,
        ):
            sample_data = self.points_pre_transform(sample_points) # ADATTATO A SHREC18
            sample_data["x"] = torch.from_numpy(sample_data["x"])
            sample_data["y"] = torch.LongTensor(
                sample_data["y"]
            )  # Need input classification for DropPointsByClass
            sample_data["pos"] = torch.from_numpy(sample_data["pos"])
            # for final interpolation - should be kept as a np.ndarray to be batched as a list later.
            sample_data["idx_in_original_cloud"] = idx_in_original_cloud
            sample_data["normals"] = torch.from_numpy(sample_data["normals"])

            # COPY OG DATA
            #sample_data["copies"] = {"x_copy": sample_data["x"], "y_copy": sample_data["y"]}

            if self.pre_filter and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue
            
            if self.transform:
                sample_data = self.transform(sample_data)

            if sample_data is None:
                continue

            if self.pre_filter and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue

            yield sample_data

    def get_subdivided_tile(self, tile, n_centroids):
        for idx_in_voronoi_cell in voronoi_refine(tile, n_centroids):
            sample_data = Data(
                pos=tile["pos"][idx_in_voronoi_cell],
                x=tile["x"][idx_in_voronoi_cell],
                y=tile["y"][idx_in_voronoi_cell],
                x_features_names=tile["x_features_names"],
                idx_in_original_cloud=tile["idx_in_original_cloud"][idx_in_voronoi_cell]
            )

            if self.pre_filter and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue
            
            if self.transform:
                sample_data = self.transform(sample_data)

            if sample_data is None:
                continue

            if self.pre_filter and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue

            yield sample_data