# Binary Segmentation of Relief Patterns
This repository contains the official implementation of the paper *Binary segmentation of relief patterns on point clouds* by Gabriele Paolini, Claudio Tortorici and Stefano Berretti, which presents a neural network designed to perform binary segmentation on surfaces with periodic reliefs. 

The code is largely based on the deep learning library [Myria3D](https://github.com/IGNF/myria3d).
The repository extends the Myria3D framework by implementing the 3D segmentation neural network introduced in the aforementioned paper.
Although Myria3D was initially built to process aerial lidar data, the framework is flexible enough to support generic 3D data signatures.
Currently, the framework expects the dataset either as a single [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file or as a set of folders (train/val/test) containing the point clouds in [LAS](https://en.wikipedia.org/wiki/LAS_file_format) format.

> [!NOTE]
> For more details on using the framework, please refer to the official Myria3D [Documentation](https://ignf.github.io/myria3d/).

## Repository structure
The repository contains the following items:
- **myria3d**: This directory contains the Myria3D framework along with the test data related to Fig.11 of the paper (located in the ```/myria3d_cross/tests/data``` folder), a checkpoint of the trained model (in ```/myria3d_cross/logs```) and the network architecture (see ```/myria3d_cross/myria3d/models/modules```).
- **Dockerfile**: We provide a dockerfile to generate a system-independent Python environment where training and tests can be performed. The corresponding Docker image requires almost 14GB of memory space (this includes most of the dependencies on which Myria3D is built).
- **README.md**: This file contains a basic description of the project and instructions on how to build and run an inference test.
- **LICENSE**: This is the license under which the project's code is distributed.

## How to produce Fig.11 from the paper
Assuming you have [Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, you can build the Docker image by running these instructions in the root directory of the project:

```
docker build -t bin-seg .
```

The command starts downloading the various layers that compose the Docker image.
Most libraries support the execution of [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) on the GPU.
While this is not mandatory for performing inference tests on a single point cloud, as the code can be executed on CPU, I haven't had the time to write a proper Dockerfile for CPU execution.

> [!WARNING]
> The code was mainly tested on Ubuntu 22.04.4 machines with Nvidia GPUs (GeForce GTX 1050Ti Mobile and GeForce RTX 2080) and Ubuntu 24.04 with Nvidia RTX 3050.

Once the image has been built, you can create a new temporary Docker container by running:

```
docker run --gpus all -it -v ./myria3d_cross:/app/myria3d_cross --rm bin-seg
```

The ```-it``` flag starts the container in interactive mode, while ```-v``` tells the Docker to mount the specified directory on the host inside the container (in this case, inside the working directory ```/app```).
The ```--rm``` option is used to erase everything related to the container as soon as it is stopped.

To start the inference test on the surface from Fig.11, run the following instructions inside the ```./myria3d_cross``` folder:

```
cd myria3d_cross
python run.py task.task_name=predict
```

If the execution completes successfully, you should find a file named **wand.las** inside the ```./myria3d_cross/outputs``` folder.
The point cloud can be viewed with free tools such as [CloudCompare](https://www.cloudcompare.org/).
The predicted segmentation labels are stored in the **PredictedClassification** scalar field.
Using CloudCompare, red points should denote smooth areas, while blue points belong to textured regions.
For a better view, please increase the point size.

### Troubleshooting
If the Docker image build fails or you encounter the following error:

```
RuntimeError: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW
```

This issue may be related to an incompatibility between the Nvidia driver version installed on your computer and the CUDA runtime version required by the image.
First, check the Nvidia driver version by running the ```nvidia-smi``` command.
Then visit the [CUDA Application Compatibility Support Matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#id3) from the official Nvidia documentation to determine the CUDA runtime version compatible with your Nvidia driver version.
Depending on your Nvidia driver verion, you should update the base image in your dockerfile to the first compatible CUDA runtime version.
For example, if you have Nvidia driver version 535 or higher, you need to modify the first line of your dockerfile to use ```nvidia/cuda:12.2-runtime-ubuntu22.04``` or a higher version.

If the program raises a ```KeyError```:

```
raise KeyError(f"Environment variable '{key}' not found")
omegaconf.errors.InterpolationResolutionError: KeyError raised while resolving interpolation: "Environment variable 'LOGS_DIR' not found"
    full_key: hydra.run.dir
    object_type=dict

```

export the corresponding environment variable by typing ```export LOGS_DIR=logs``` in the terminal. 

## How to cite
If you want to cite this work, please use the following BibTex entry:
```
@article{PAOLINI2024104020,
title = {Binary segmentation of relief patterns on point clouds},
journal = {Computers & Graphics},
volume = {123},
pages = {104020},
year = {2024},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2024.104020},
url = {https://www.sciencedirect.com/science/article/pii/S0097849324001559},
author = {Gabriele Paolini and Claudio Tortorici and Stefano Berretti},
keywords = {Relief pattern, Point cloud, 3D segmentation},
abstract = {Analysis of 3D textures, also known as relief patterns is a challenging task that requires separating repetitive surface patterns from the underlying global geometry. Existing works classify entire surfaces based on one or a few patterns by extracting ad-hoc statistical properties. Unfortunately, these methods are not suitable for objects with multiple geometric textures and perform poorly on more complex shapes. In this paper, we propose a neural network for binary segmentation to infer per-point labels based on the presence of surface relief patterns. We evaluated the proposed architecture on a high resolution point cloud dataset, surpassing the state-of-the-art, while maintaining memory and computation efficiency.}
}
```
