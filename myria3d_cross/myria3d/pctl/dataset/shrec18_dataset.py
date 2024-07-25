"""Generation of a toy dataset for testing purposes."""

import os
import os.path as osp
import sys

# to use from CLI.
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))
from myria3d.pctl.dataset.hdf5 import HDF5Dataset  # noqa
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform


basedir = "tests/data"
dirname = "crossvalidation"
SHREC18_EPSG = "2154"
SHREC18_LAS_DATA_TRAIN = f"{basedir}/{dirname}/train/"
SHREC18_LAS_DATA_TEST = f"{basedir}/{dirname}/test/"
SHREC18_LAS_DATA_VAL = f"{basedir}/{dirname}/val/"
SHREC18_DATASET_HDF5_PATH = f"{basedir}/{dirname}.hdf5"


def make_toy_dataset_from_test_file():
    """Prepare a toy dataset from a single, small LAS file.

    The file is first duplicated to get 2 LAS in each split (train/val/test),
    and then each file is splitted into .data files, resulting in a training-ready
    dataset loacted in td_prepared

    Args:
        src_las_path (str): input, small LAS file to generate toy dataset from
        split_csv (str): Path to csv with a `basename` (e.g. '123_456.las') and
        a `split` (train/val/test) columns specifying the dataset split.
        prepared_data_dir (str): where to copy files (`raw` subfolder) and to prepare
        dataset files (`prepared` subfolder)

    Returns:
        str: path to directory containing prepared dataset.

    """
    if os.path.isfile(SHREC18_DATASET_HDF5_PATH):
        os.remove(SHREC18_DATASET_HDF5_PATH)

    # TODO: update transforms ? or use a config ?
    HDF5Dataset(
        SHREC18_DATASET_HDF5_PATH,
        SHREC18_EPSG,
        data_paths_by_split_dict={
            "train": [SHREC18_LAS_DATA_TRAIN + f for f in os.listdir(SHREC18_LAS_DATA_TRAIN)],
            "test": [SHREC18_LAS_DATA_TEST + f for f in os.listdir(SHREC18_LAS_DATA_TEST)],
            "val": [SHREC18_LAS_DATA_VAL + f for f in os.listdir(SHREC18_LAS_DATA_VAL)],
        },
        tile_width=100, #110
        subtile_width=50,
        train_transform=None,
        eval_transform=None,
        pre_filter=None,
        points_pre_transform=lidar_hd_pre_transform,
        file_format="las",
        sampling_method="voronoi",
    )
    return SHREC18_DATASET_HDF5_PATH


if __name__ == "__main__":
    make_toy_dataset_from_test_file()
