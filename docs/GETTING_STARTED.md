# Getting Started
The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 


## Dataset Preparation

Currently we provide the dataloader of KITTI dataset and NuScenes dataset, and the supporting of more datasets are on the way.  

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to train [SGM3D](../tools/cfgs/kitti_models/sgm3d.yaml), download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) and [pseudo-lidar_velodyne](https://drive.google.com/file/d/10txZOtKk_aY3B7AhHjJPMCiRf5pP62nV/view?usp=sharing) for the KITTI training set
* NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged.

```
sgm3d
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib
│   │   │   ├──image_2
│   │   │   ├──depth_2
│   │   │   ├──label_2
│   │   │   ├──planes
│   │   │   |──velodyne --> pseudo-liear_velodyne
│   │   │── testing
│   │   │   ├──calib
│   │   │   ├──image_2
│   │   │   |──velodyne
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```



## Pretrained Models
If you would like to train [SGM3D](../tools/cfgs/kitti_models/sgm3d.yaml), download the pretrained [DeepLabV3 model](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth), [distilled_ppc_stereo_80](https://drive.google.com/file/d/1K7SimAD2eNObHtHkG_Irc-V6SjFQVcE2/view?usp=sharing) and place within the `checkpoints` directory
```
OpenPCDet
├── checkpoints
│   ├── deeplabv3_resnet101_coco-586e9e4e.pth
    ├── distilled_ppc_stereo_80.pth
├── data
├── pcdet
├── tools
```

## Training & Testing


### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 

If you would like to train [SGM3D](../tools/cfgs/kitti_models/sgm3d.yaml), use `--pretrained_models ../checkpoints/distilled_ppc_stereo_80.pth` to load the pretrained stereo model.

* Train with multiple GPUs or multiple machines
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
