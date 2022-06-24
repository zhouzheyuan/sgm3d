# SGM3D

**SGM3D: Stereo Guided Monocular 3D Object Detection**
Zheyuan Zhou, Liang Du, Xiaoqing Ye, Zhikang Zou, Xiao Tan, Li Zhang, Xiangyang Xue, Jianfeng Feng
[`[Paper]`](https://arxiv.org/pdf/2112.01914.pdf).
<!-- ![demo](docs/demo.gif) -->


## Overview
- [Changelog](#changelog)
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)


## Changelog

[2021-06-24] `SGM3D` is released. 


## Model Zoo

### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance on the *val* set of KITTI dataset.

|                                             | Easy@R40 | Mod.@R40 | Hard@R40 | download | 
|:-------------------------------------------:|:--------:|:--------:|:--------:|:--------:|
| [SGM3D](tools/cfgs/kitti_models/sgm3d.yaml) | 25.96 | 17.81 | 15.11 | [model-874M]() |


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


## Acknowledgements
'SGM3D' is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 


## Citation 
If you find this project useful in your research, please consider cite:


```bibtex
@article{zhou2021sgm3d,
  title={SGM3D: Stereo Guided Monocular 3D Object Detection},
  author={Zhou, Zheyuan and Du, Liang and Ye, Xiaoqing and Zou, Zhikang and Tan, Xiao and Zhang, Li and Xue, Xiangyang and Feng, Jianfeng},
  journal={arXiv},
  year={2021}
}
```




