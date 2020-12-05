# PointMixup: Augmentation for point cloud

This repository contains an implementation to the ECCV 2020 paper: "PointMixup: Augmentation for point cloud".

### Install
============================
- python3.6, pytorch 1.2, tensorboardX
- pip install git+git://github.com/erikwijmans/etw_pytorch_utils.git@v1.1.1#egg=etw_pytorch_utils
- pip install --no-cache --upgrade git+https://github.com/dongzhuoyao/pytorchgo.git
- sh setup.sh
============================

### Run

Baseline:
```
python main.py -savename exp0 -pointmixup False -manimixup False
```

Our method (point mixup):
```
python main.py -savename exp1 -pointmixup True -manimixup False -mixup_alpha 0.4
```

Our method (point mixup + manifold mixup):
```
python main.py -savename exp2 -pointmixup True -manimixup True -mixup_alpha 1.5
```
### Data
If data cannot be downloaded automatically from the script, please manually download from 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip' and extract to './pointnet2/data/' folder.

### acknowledgement
* [Our code is based on this repo with implementation of PointNet++ in PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
* [Fast approximation of optimal assignment is adapted from this repo](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd)


