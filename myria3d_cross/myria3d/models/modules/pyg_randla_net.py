import os.path as osp
from numbers import Number
from typing import Tuple

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import LongTensor, Tensor
from torch.nn import Linear
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn_graph, radius_graph
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

import laspy
import numpy as np
#from umap import UMAP
import time
from torch_scatter import scatter_max

# Per knn interpolate with attention
from torch_geometric.utils import scatter as scatter_sum
from torch_geometric.nn import knn
from torch_geometric.typing import OptTensor
#from utilities.global_feat_extractor import global_transform, get_graph_feature

class ArrayDescriptor:
    def __init__(self, array, data_type):
        # Initialize the dictionary with a numpy array and its data type 
        self.descriptor = {
            "value": array,
            "type": data_type
        }

    def __getitem__(self, key):
        # Delegate dictionary item access to self.descriptor
        return self.descriptor[key]

class PyGRandLANet(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,           # TODO: generalizzare a un numero di classi indefinito
        global_feat_train: bool,
        decimation: int = 4,        # Fattore di decimazione della point cloud
        num_neighbors: int = 16,    # Numero di primi vicini che definiscono il campo recettivo
        return_logits: bool = False,
    ):
        super().__init__()

        self.decimation = decimation
        # An option to return logits instead of probas
        self.return_logits = return_logits

        self.global_feat_train = global_feat_train

        self.epoch = 0
        self.point_cloud_index = 0

        # Authors use 8, which is a bottleneck
        # for the final MLP, and also when num_classes>8
        # or num_features>8.
        d_bottleneck = max(16, num_classes, num_features) # prev 32

        self.fc0 = Linear(num_features, d_bottleneck)
        self.block1 = DilatedResidualBlock(num_neighbors, d_bottleneck, 32, id=0)
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128, id=1)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 256, id=2)
        self.block4 = DilatedResidualBlock(num_neighbors, 256, 256, id=3)
        self.mlp_summit = SharedMLP([256, 256]) # prev 512
        self.fp4 = AttentiveFPModule(k=8, nn=SharedMLP([256 + 256, 256]), channels=256)
        self.fp3 = AttentiveFPModule(k=16, nn=SharedMLP([256 + 128, 128]), channels=256)
        self.fp2 = AttentiveFPModule(k=8, nn=SharedMLP([128 + 32, 32]), channels=128)
        self.fp1 = FPModule(k=1, nn=SharedMLP([32 + 32, d_bottleneck]))
        self.mlp_classif = SharedMLP([d_bottleneck, 64, 32], dropout=[0.0, 0.5])
        self.fc_classif = Linear(32, num_classes)

    def save_patch(self, data, filename):
        filename = filename + str(self.point_cloud_index) +".las"
        self.point_cloud_index += 1
        if osp.exists(filename):
            return
        if self.epoch % 10 != 0:
            return 
        x, pos, batch = data[0:3]
        att_scores = data[-1]
        # Number of point clouds    

        # Indexing each point cloud (only first point cloud)
        x = x[batch == 1]    #[x[batch == i] for i in range(num_clouds)]
        pos = pos[batch == 1]    #[pos[batch == i] for i in range(num_clouds)]

        pointrecord = laspy.create(file_version="1.4", point_format=3)
        vertices = pos.detach().cpu().numpy()
        pointrecord.header.offsets = np.min(vertices, axis=0)
        point_count = vertices.shape[0]
        pointrecord.header.point_count = point_count
        pointrecord.x = vertices[:, 0]
        pointrecord.y = vertices[:, 1]
        pointrecord.z = vertices[:, 2]

        """
        # Umap
        feat = x.detach().cpu().numpy()
        if feat.shape[-1] > 2:
            start_time = time.time()
            reducer = UMAP(n_neighbors=40, n_components=3, metric='euclidean')
            data_embedded = reducer.fit_transform(feat)
            end_time = time.time()
            print("Execution time: {:.2f} seconds".format(end_time - start_time))
            pointrecord.red = data_embedded[:,0]
            pointrecord.green = data_embedded[:,1]
            pointrecord.blue = data_embedded[:,2]

            for dim in range(data_embedded.shape[-1]):
                name = f"UMAP_{dim}"
                pointrecord.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
                pointrecord[name] = data_embedded[:, dim]
        else:
            for dim in range(x.size(-1)):    
                name = f"Feature_{dim}"
                pointrecord.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
                pointrecord[name] = feat[:, dim]
        """

        # Salvataggio di tutto il vettore di features
        feat = x.detach().cpu().numpy()
        for feature_index in range(feat.shape[-1]):
            name = f"Feature_{feature_index}"
            pointrecord.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
            pointrecord[name] = feat[:, feature_index]
        pointrecord.write(filename)
        print(f"File written in {filename}")

    def update_epoch(self):
        self.epoch += 1
        self.block1.update_epoch()
        self.block2.update_epoch()
        self.block3.update_epoch()
        self.block4.update_epoch()

    def forward(self, x, pos, batch, ptr, normals):
        x = x.float() if x is not None else pos # CAMBIATO x da float64 a float32

        # Global feat extraction
        #PV = global_transform(pos, npoints=31, train=self.global_feat_train) # knn=32
        #self.save_patch((x, pos, batch), f"patch_input")

        b1_out = self.block1(self.fc0(x), pos, batch, normals)
        #self.save_patch(b1_out, f"patch_b1")
        b1_out_decimated, ptr1 = decimate(b1_out, ptr, self.decimation)

        b2_out = self.block2(*b1_out_decimated)
        #self.save_patch(b2_out, f"patch_b2")
        b2_out_decimated, ptr2 = decimate(b2_out, ptr1, self.decimation)

        b3_out = self.block3(*b2_out_decimated)
        #self.save_patch(b3_out, f"patch_b3")
        b3_out_decimated, ptr3 = decimate(b3_out, ptr2, self.decimation)

        b4_out = self.block4(*b3_out_decimated)
        #self.save_patch(b4_out, f"patch_b4")
        b4_out_decimated, _ = decimate(b4_out, ptr3, self.decimation)

        mlp_out = ( # Da cambiare
            self.mlp_summit(b4_out_decimated[0]),
            b4_out_decimated[1],
            b4_out_decimated[2],
        )
        #self.save_patch(mlp_out, f"patch_summit")

        fp4_out = self.fp4(*mlp_out, *b3_out_decimated[:-1])    # -2 perchè c'è anche PV
        #self.save_patch(fp4_out, f"patch_fp4")
        fp3_out = self.fp3(*fp4_out, *b2_out_decimated[:-1])
        #self.save_patch(fp3_out, f"patch_fp3")
        fp2_out = self.fp2(*fp3_out, *b1_out_decimated[:-1])
        #self.save_patch(fp2_out, f"patch_fp2")
        fp1_out = self.fp1(*fp2_out, *b1_out[:-1])
        #self.save_patch(fp1_out, f"patch_fp1")

        x = self.mlp_classif(fp1_out[0])
        #self.save_patch((x, pos, batch), f"patch_out")
        logits = self.fc_classif(x)
        #self.save_patch((logits, pos, batch), f"logits")

        if self.return_logits:
            return logits

        probas = logits.log_softmax(dim=-1)
        return probas


# Default activation, BatchNorm, and resulting MLP used by RandLA-Net authors
lrelu02_kwargs = {"negative_slope": 0.2}

bn099_kwargs = {"momentum": 0.01, "eps": 1e-6}


class SharedMLP(MLP):
    """SharedMLP following RandLA-Net paper."""

    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs["plain_last"] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs["act"] = kwargs.get("act", "LeakyReLU")
        kwargs["act_kwargs"] = kwargs.get("act_kwargs", lrelu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", bn099_kwargs)
        super().__init__(*args, **kwargs)

class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, channels):
        super().__init__(aggr="add")
        self.mlp_encoder = SharedMLP([8, channels // 2]) # Con RI è 8, con pos è 10
        #self.mlp_global_encoder = SharedMLP([6, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False, act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])
        #self.att_scores_max = None # DEBUG VISUALIZZAZIONE

    def forward(self, edge_index, x, pos, normals):
        out = self.propagate(edge_index, x=x, pos=pos, normals=normals)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, index: Tensor, 
                normals_j: Tensor, normals_i: Tensor, edge_index: Tensor) -> Tensor:
        """Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        # Encode local neighboorhod structural information
        relative_infos = torch.empty([pos_i.size(0), 8], device=pos_i.device)
        relative_infos = self.darboux(pos_i, pos_j, normals_i, normals_j, feature=relative_infos)
        #pos_diff = pos_j - pos_i
        #distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        #relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance], dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d

        # Global features
        #global_infos = torch.cat([PV_j, PV_i - PV_j], dim=1) 
        #global_spatial_encoding = self.mlp_global_encoder(global_infos)

        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N * K, 2d 

        # Attention will weight the different features of x
        # along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out   # TODO: RICALCOLA OGNI VOLTA GLI ATT_FEATURES???
        att_scores = softmax(att_features, index=index)  # N * K, d_out == N*K,2d

        # Tensore per il salvataggio dei valori di attention
        #self.att_scores_max = scatter_max(src=att_scores, index=edge_index[0], dim=0)[0]  # Forse la media?

        # NON E' IL PRODOTTO SCALARE MA UN ELEMENT-WISE PRODUCT!!!
        return att_scores * local_features  # N * K, d_out 

    def darboux(self, pos_i, pos_j, normals_i, normals_j, feature):
        pos_diff = pos_j - pos_i
        
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))   # Prima feature
        feature[:,0] = distance.squeeze()
        len_normals_i = torch.norm(normals_i, dim=1).unsqueeze(-1)
        len_normals_j = torch.norm(normals_j, dim=1).unsqueeze(-1)

        feature[:,1] = (torch.sum(pos_diff * normals_i, dim=1, keepdim=True) / (distance * len_normals_i + 1e-10)).squeeze()
        feature[:,2] = (torch.sum(pos_diff * normals_j, dim=1, keepdim=True) / (distance * len_normals_j + 1e-10)).squeeze()
        feature[:,3] = (torch.sum(normals_i * normals_j, dim=1, keepdim=True) / (len_normals_i * len_normals_j + 1e-10)).squeeze()

        uq = torch.cross(pos_diff, normals_i, dim=1)
        vq = torch.cross(uq, normals_i, dim=1)
        uk = torch.cross(pos_diff, normals_j, dim=1)
        vk = torch.cross(uk, normals_j, dim=1)

        len_uq = torch.norm(uq, dim=1).unsqueeze(-1)
        len_vq = torch.norm(vq, dim=1).unsqueeze(-1)
        len_uk = torch.norm(uk, dim=1).unsqueeze(-1)
        len_vk = torch.norm(vk, dim=1).unsqueeze(-1)

        feature[:,4] = (torch.sum(uq * uk, dim=1, keepdim=True) / (len_uq * len_uk + 1e-10)).squeeze()
        feature[:,5] = (torch.sum(vq * vk, dim=1, keepdim=True) / (len_vq * len_vk + 1e-10)).squeeze()
        feature[:,6] = (torch.sum(uq * vk, dim=1, keepdim=True) / (len_uq * len_vk + 1e-10)).squeeze()
        feature[:,7] = (torch.sum(vq * uk, dim=1, keepdim=True) / (len_vq * len_uk + 1e-10)).squeeze()

        #return pos_diff, distance
        return feature

class DilatedResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_neighbors,
        d_in: int,
        d_out: int,
        id: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out

        # Dilated knn
        #self.k_range = [4,8]
        #self.d_range = [1,4]

        # Distance-based knn
        #self.radius = 3

        #self.epoch = 0
        self.id = id

        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)  # Skip connection MLP
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch, normals):
        # Dilated KNN
        #k = torch.randint(self.k_range[0], self.k_range[1] + 1, (1,))
        #d = torch.randint(self.d_range[0], self.d_range[1] + 1, (1,))
        #span = k*d
        #print(f"K: {k}, D: {d}, Span: {span}")

        # Funzione che ritorna i K primi vicini per ogni punto della point cloud
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True) 
        #edge_index = radius_graph(x=pos, r=self.radius, batch=batch, loop=True, max_num_neighbors=self.num_neighbors)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos, normals)  # N, d_out//2
        
        x = self.lfa2(edge_index, x, pos, normals)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        #self.save_attention_map([self.lfa1.att_scores_max, self.lfa2.att_scores_max], pos, batch, f"Patch_B{self.id}")

        return x, pos, batch, normals
    
    def save_attention_map(self, maps, pos, batch, filename):
        if self.epoch % 10 != 0:
            return   

        # Indexing each point cloud (all point clouds)
        num_clouds = torch.unique(batch).size(0)
        for sample in range(num_clouds):

            filename = filename + "_S" + str(sample) + "_E" + str(self.epoch) + ".las"
            if osp.exists(filename):
                return
            
            maps_sample = [maps[layer][batch == sample] for layer in range(len(maps))]
            pos_sample = pos[batch == sample]    #[pos[batch == i] for i in range(num_clouds)]

            pointrecord = laspy.create(file_version="1.4", point_format=3)
            vertices = pos_sample.detach().cpu().numpy()
            pointrecord.header.offsets = np.min(vertices, axis=0)
            point_count = vertices.shape[0]
            pointrecord.header.point_count = point_count
            pointrecord.x = vertices[:, 0]
            pointrecord.y = vertices[:, 1]
            pointrecord.z = vertices[:, 2]

            for layer in range(len(maps_sample)):
                for dim in range(maps_sample[layer].size(-1)):    
                    name = f"Att_L{layer}_D{dim}"
                    pointrecord.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))
                    pointrecord[name] = maps_sample[layer][:, dim]
            pointrecord.write(filename)

    def update_epoch(self):
        self.epoch += 1

def decimation_indices(ptr: LongTensor, decimation_factor: Number) -> Tuple[Tensor, LongTensor]:
    """Get indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have a least
    one node after decimation.

    Args:
        ptr (LongTensor): indices of samples in the batch.
        decimation_factor (Number): value to divide number of nodes with.
            Should be higher than 1 for downsampling.

    :rtype: (:class:`Tensor`, :class:`LongTensor`): indices for downsampling
        and resulting updated ptr.

    """
    if decimation_factor < 1:
        raise ValueError(
            "Argument `decimation_factor` should be higher than (or equal to) "
            f"1 for downsampling. (Current value: {decimation_factor})"
        )

    batch_size = ptr.size(0) - 1
    bincount = ptr[1:] - ptr[:-1]
    decimated_bincount = torch.div(bincount, decimation_factor, rounding_mode="floor")
    # Decimation should not empty clouds completely.
    decimated_bincount = torch.max(torch.ones_like(decimated_bincount), decimated_bincount)
    idx_decim = torch.cat(
        [
            (ptr[i] + torch.randperm(bincount[i], device=ptr.device)[: decimated_bincount[i]])
            for i in range(batch_size)
        ],
        dim=0,
    )
    # Get updated ptr (e.g. for future decimations)
    ptr_decim = ptr.clone()
    for i in range(batch_size):
        ptr_decim[i + 1] = ptr_decim[i] + decimated_bincount[i]

    return idx_decim, ptr_decim

def decimation_indices_fps(ptr: LongTensor, decimation_factor: Number, tensors) -> Tuple[Tensor, LongTensor]:
    if decimation_factor < 1:
        raise ValueError(
            "Argument `decimation_factor` should be higher than (or equal to) "
            f"1 for downsampling. (Current value: {decimation_factor})"
        )

    batch_size = ptr.size(0) - 1
    bincount = ptr[1:] - ptr[:-1]
    decimated_bincount = torch.div(bincount, decimation_factor, rounding_mode="floor")
    # Decimation should not empty clouds completely.
    decimated_bincount = torch.max(torch.ones_like(decimated_bincount), decimated_bincount)
    # FPS 
    decimated_indices = []
    for i in range(batch_size):
        start_idx = torch.randint(high=bincount[i], size=(1,), device=ptr.device)
        indices = [start_idx.item()]
        pos = tensors[1][ptr[i]:ptr[i]+bincount[i]]
        min_distances = torch.full((bincount[i],), float('inf'), device=ptr.device)
        
        for _ in range(int(decimated_bincount[i]) - 1):
            cur_point = pos[indices[-1]]
            dist_to_cur_point = torch.norm(pos - cur_point.unsqueeze(0), dim=1)
            min_distances = torch.minimum(min_distances, dist_to_cur_point)
            farthest_point_idx = torch.argmax(min_distances).item()
            indices.append(farthest_point_idx)
        decimated_indices.extend(ptr[i] + torch.tensor(indices, device=ptr.device))
    idx_decim = torch.tensor(decimated_indices, device=ptr.device)
    
    # Get updated ptr (e.g. for future decimations)
    ptr_decim = ptr.clone()
    for i in range(batch_size):
        ptr_decim[i + 1] = ptr_decim[i] + decimated_bincount[i]

    return idx_decim, ptr_decim

def decimate(tensors, ptr: Tensor, decimation_factor: int):
    """Decimate each element of the given tuple of tensors."""
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)    #, tensors)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim

class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class AttentiveFPModule(torch.nn.Module):
    """ Upsampling with skip connection/attention mechanism/multiple neighbours"""
    def __init__(self, k, nn, channels):
        super().__init__()
        self.k = k
        self.mlp_attention = SharedMLP([channels, 1], bias=False, act=None, norm=None)
        self.nn = nn
    
    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = self.knn_att_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)  # k == 1 (renderlo randomico? Con dilazione?)
        x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
    
    def knn_att_interpolate(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3, num_workers: int = 1):
        with torch.no_grad():
            assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y,
                            num_workers=num_workers)
            y_idx, x_idx = assign_index[0], assign_index[1]
            # Calcolo distanze (euclidee)
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        # Calcolo attention scores BUG 31/03 GLI ATTENTION SCORES VENIVANO CALCOLATI DENTRO NO_GRAD()!!!
        att_features = self.mlp_attention(x)[x_idx]
        att_scores = softmax(src=att_features, index=y_idx)

        y = scatter_sum(x[x_idx] * weights * att_scores, y_idx, 0, pos_y.size(0), reduce='sum')
        y = y / scatter_sum(weights, y_idx, 0, pos_y.size(0), reduce='sum')

        return y


def main():
    category = "Airplane"  # Pass in `None` to train on all categories.
    category_num_classes = 4  # 4 for Airplane - see ShapeNet for details
    path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "..",
        "..",
        "..",
        "data",
        "ShapeNet",
    )
    transform = T.Compose(
        [
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
        ]
    )
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(
        path,
        category,
        split="trainval",
        transform=transform,
        pre_transform=pre_transform,
    )
    test_dataset = ShapeNet(path, category, split="test", pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyGRandLANet(3, category_num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()

        total_loss = correct_nodes = total_nodes = 0
        for i, data in tqdm(enumerate(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.pos, data.batch, data.ptr)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes

            if (i + 1) % 10 == 0:
                print(
                    f"[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} "
                    f"Train Acc: {correct_nodes / total_nodes:.4f}"
                )
                total_loss = correct_nodes = total_nodes = 0

    @torch.no_grad()
    def test(loader):
        model.eval()

        ious, categories = [], []
        y_map = torch.empty(loader.dataset.num_classes, device=device).long()
        for data in loader:
            data = data.to(device)
            outs = model(data.x, data.pos, data.batch, data.ptr)

            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
            for out, y, category in zip(
                outs.split(sizes), data.y.split(sizes), data.category.tolist()
            ):
                category = list(ShapeNet.seg_classes.keys())[category]
                part = ShapeNet.seg_classes[category]
                part = torch.tensor(part, device=device)

                y_map[part] = torch.arange(part.size(0), device=device)

                iou = jaccard_index(
                    out[:, part].argmax(dim=-1),
                    y_map[y],
                    num_classes=part.size(0),
                    absent_score=1.0,
                )
                ious.append(iou)

            categories.append(data.category)

        iou = torch.tensor(ious, device=device)
        category = torch.cat(categories, dim=0)

        mean_iou = scatter(iou, category, reduce="mean")  # Per-category IoU.
        return float(mean_iou.mean())  # Global IoU.

    for epoch in range(1, 31):
        train()
        iou = test(test_loader)
        print(f"Epoch: {epoch:02d}, Test IoU: {iou:.4f}")


if __name__ == "__main__":
    main()
