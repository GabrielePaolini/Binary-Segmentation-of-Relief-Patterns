# Binary Segmentation of Relief Patterns
This repository contains the official implementation of the paper *Binary segmentation of relief patterns on point clouds*, which presents a neural network designed to perform binary segmentation on surfaces with periodic reliefs. 

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
Assuming you have [Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/) installed, you can build the Docker image by running these instructions in the root directory of the project:

```
docker build -t bin-seg .
```

The command starts downloading the various layers that compose the Docker image.
Most libraries support the execution of [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) on the GPU.
While this is not mandatory for performing inference tests on a single point cloud, as the code can be executed on CPU, I haven't had the time to write a proper Dockerfile for CPU execution.

> [!WARNING]
> The code was tested on Nvidia GPUs (GeForce GTX 1050Ti Mobile and GeForce RTX 2080).

Once the image has been built, you can create a new temporary Docker container by running:

```
docker run --gpus all -it -v ./myria3d_cross:/app/myria3d_cross --rm bin-seg
```

The ```-it``` flag starts the container in interactive mode, while ```-v``` tells the Docker to mount the specified directory on the host inside the container (in this case, inside the working directory ```/app```).
The ```--rm``` option is used to erase everything related to the container as soon as it is stopped.

To start the inference test on the surface from Fig.11, run the following instructions inside the ```/myria3d_cross``` folder:

```
python run.py task.task_name=predict
```

If the execution completes successfully, you should find a file named **wand.las** inside the ```/myria3d_cross/outputs``` folder.
The point cloud can be viewed with free tools such as [CloudCompare](https://www.cloudcompare.org/).
The predicted segmentation labels are stored in the **PredictedClassification** scalar field.
Using CloudCompare, red points should denote smooth areas, while blue points belong to textured regions.
For a better view, please increase the point size.

### Troubleshooting
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
BibTex entry (WIP)
```
