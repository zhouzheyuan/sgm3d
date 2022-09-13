# SGM3D

> [**SGM3D: Stereo Guided Monocular 3D Object Detection**](https://ieeexplore.ieee.org/document/9832729),  
> Zheyuan Zhou, Liang Du, Xiaoqing Ye, Zhikang Zou, Xiao Tan, Li Zhang, Xiangyang Xue, Jianfeng Feng  
> **RA-L 2022**


![demo](docs/sgm3d_demo.gif)


## Overview
- [Changelog](#changelog)
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)


## Changelog

[2022-07-14] Paper is received by [RA-L](https://ieeexplore.ieee.org/document/9832729).

[2022-06-24] Code is released.

[2021-12-03] Paper is released on [Arxiv](https://arxiv.org/pdf/2112.01914.pdf).

## Model Zoo

### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance of car class on the *val* set of KITTI dataset.

|                                             | Easy@R40 | Mod.@R40 | Hard@R40 | download | 
|:-------------------------------------------:|:--------:|:--------:|:--------:|:--------:|
| [SGM3D](tools/cfgs/kitti_models/sgm3d.yaml) | 25.95 | 17.44 | 15.36 | [model-292M](https://drive.google.com/file/d/13oAMRWOfqakuCmoKvZ6duFO12qsQI3Ii/view?usp=sharing) |


## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


## Acknowledgements
`SGM3D` is based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 


## Citation 
If you find this project useful in your research, please consider cite:


```bibtex
@article{zhou2022sgm3d,
  title={SGM3D: Stereo Guided Monocular 3D Object Detection}, 
  author={Zhou, Zheyuan and Du, Liang and Ye, Xiaoqing and Zou, Zhikang and Tan, Xiao and Zhang, Li and Xue, Xiangyang and Feng, Jianfeng},
  journal={IEEE Robotics and Automation Letters}, 
  year={2022},
  volume={7},
  number={4},
  pages={10478-10485},
  doi={10.1109/LRA.2022.3191849}}
```




