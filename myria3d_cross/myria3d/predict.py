import os
import os.path as osp
import sys

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from myria3d.models.model import Model

sys.path.append(osp.dirname(osp.dirname(__file__)))
from myria3d.models.interpolation import Interpolator  # noqa
from myria3d.utils import utils  # noqa

from torch_geometric.data import Batch

log = utils.get_logger(__name__)


@utils.eval_time
def predict(config: DictConfig) -> str:
    """
    Inference pipeline.

    A lightning datamodule splits a single point cloud of arbitrary size (typically: 1km * 1km) into subtiles
    (typically 50m * 50m), which are grouped into batches that are fed to a trained neural network embedded into a lightning Module.

    Predictions happen on a subsampled version of each subtile, which needs to be propagated back to the complete
    point cloud via an Interpolator. This Interpolator also includes the creation of a new LAS/OBJ file with additional
    dimensions, including predicted classification, entropy, and (optionnaly) predicted probability for each class.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        str: path to ouptut LAS/OBJ.

    """

    # Those are the 2 needed inputs, in addition to the hydra config.
    assert os.path.exists(config.predict.ckpt_path)
    assert os.path.exists(config.predict.src_data)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data(config.predict.src_data)

    # Do not require gradient for faster predictions
    torch.set_grad_enabled(False)
    model = Model.load_from_checkpoint(config.predict.ckpt_path, strict=False)
    model.criterion = hydra.utils.instantiate(config.model.criterion)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    # TODO: Interpolator could be instantiated directly via hydra.
    itp = Interpolator(
        interpolation_k=config.predict.interpolator.interpolation_k,
        classification_dict=config.dataset_description.get("classification_dict"),
        probas_to_save=config.predict.interpolator.probas_to_save,
        predicted_classification_channel=config.predict.interpolator.get(
            "predicted_classification_channel", "PredictedClassification"
        ),
        entropy_channel=config.predict.interpolator.get("entropy_channel", "entropy"),
        file_format=config.datamodule.file_format,
        output_file_format="ply"
    )

    #import numpy as np
    #from mayavi import mlab
    #from mayavi import mlab
    
    tileidx = 0
    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)
        out = model.predict_step(batch)
        logits = out["logits"]

        # itp.idx_in_full_cloud_list è una lista di array contenente ciascuno gli indici che fanno riferimento ai nodi nella cloud originale!
        itp.store_predictions(logits, batch.idx_in_original_cloud)


        # Questo processo va ripetuto finchè:
        # 1) Non vi sono più tiles con entropia alta;
        # 2) Non vi sono più tiles processabili.
        # Check entropy values for each subtiles
        #high_entropy_subtiles = itp.check_entropy(config.predict.src_data, logits, batch.idx_in_original_cloud)
        #high_entropy_subtiles = list(range(batch.batch_size))
        #print(high_entropy_subtiles)

        """
        # Test su singola tile
        print(high_entropy_subtiles)
        unbatched_list = batch.to_data_list() 
        chosen_tile = unbatched_list[high_entropy_subtiles[0]]
        print(chosen_tile)
        batch = Batch.from_data_list([chosen_tile])
        batch.to(device)
        n_reps = 10
        for i in range(n_reps):      
            out = model.predict_step(batch)
            logits = out["logits"]
            preds = itp.compute_predictions(logits)

            pos = np.asarray(chosen_tile["pos"], dtype=np.float32).transpose()
            values = np.zeros((pos.shape[0], 3))
            print(len(preds))
            print(len(pos))
            idx = np.where(preds == 0)[0]
            values[idx] = np.repeat(np.array([1,0,0]).transpose(), values.shape[0], axis=0)
            idx = np.where(preds == 1)[0]
            values[preds == 1] = np.repeat(np.array([0,0,1]).transpose(), values.shape[0], axis=0)
            mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], color=values, scale_factor=1)
            mlab.show()
            # itp.idx_in_full_cloud_list è una lista di array contenente ciascuno gli indici che fanno riferimento ai nodi nella cloud originale!
            itp.store_predictions(logits, batch.idx_in_original_cloud)
        """
        """
        max_it = 4
        it = 0
        while high_entropy_subtiles != [] and it <= max_it:
            it += 1
            # Create a list of Data objects with low entropy
            #data_list = batch.to_data_list()
            unbatched_list = batch.to_data_list()
            data_list = []

            # Replace subtile with subdivision
            for id in high_entropy_subtiles:
                for sample_data in datamodule.predict_dataset.get_subdivided_tile(
                    tile=unbatched_list[id],
                    n_centroids=3,  
                ):
                    data_list.append(sample_data)
                #del data_list[id]
            if data_list != []:
                batch = Batch.from_data_list(data_list)
                batch.to(device)
                out = model.predict_step(batch)
                logits = out["logits"]

                # itp.idx_in_full_cloud_list è una lista di array contenente ciascuno gli indici che fanno riferimento ai nodi nella cloud originale!
                itp.store_predictions(logits, batch.idx_in_original_cloud)

                high_entropy_subtiles = itp.check_entropy(
                    config.predict.src_data, logits, batch.idx_in_original_cloud
                )
            else:
                break
            """
     
    # L'interpolazione tramite IDW avviene a questo punto!
    out_f = itp.reduce_predictions_and_save(
        config.predict.src_data, config.predict.output_dir, config.datamodule.get("epsg")
    )
    return out_f

    """
    # DA INSERIRE SUBITO DOPO LA DEFINIZIONE DI ITP
    # PLOTTING SUBTILES
    import laspy
    import numpy as np
    from torch_geometric.data import Data
    from myria3d.pctl.dataset.utils import split_cloud_into_samples
    from mayavi import mlab
    import random
    
    # Setting up the plot
    for idx_in_original_cloud, sample_points in split_cloud_into_samples(
        config.predict.src_data,
        config.datamodule.tile_width,
        config.datamodule.subtile_width,
        config.datamodule.epsg,
        config.predict.subtile_overlap,
    ):
        pos = np.asarray([sample_points["X"], sample_points["Y"], sample_points["Z"]], dtype=np.float32).transpose()
        src_data = laspy.read(config.predict.src_data)
        curvature = src_data.points["Curvature"][idx_in_original_cloud]
        x = np.stack([curvature], axis=0,).transpose()
        x_features_names = ["Curvature",]
        y = sample_points["Classification"]

        data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names)
        values = (random.random(), random.random(), random.random())
        mlab.points3d(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2], color=values, scale_factor=1)
    mlab.show()
    """