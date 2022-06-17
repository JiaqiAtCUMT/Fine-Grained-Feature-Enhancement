# [Fine-Grained Feature Enhancement for Object Detection in Remote Sensing Images](https://ieeexplore.ieee.org/document/9744034)

Yong Zhou , Sifan Wang, Jiaqi Zhao, Hancheng Zhu , and Rui Yao

## Introudction

Recently, object detection in aerial images has ushered in a new challenge—a new benchmark for fine-grained object recognition in high-resolution remote sensing imagery called FAIR1M has been proposed. Fine-grained categories usually have smaller inter class differences and intra-class similarities, which is more difficult to classify with existing object detectors. To address this problem, we propose two enhanced strategies on the current two-stage object detection algorithm. The first strategy uses attention-based group feature enhancement called group enhance module (GEM). By extending and grouping feature channels, the model can improve the ability to extract various discriminative features. The second strategy is to emphasize the sub-saliency feature learning, avoiding the network only focusing on the most significant part of the feature and ignoring the other parts. Our method is easy to implement and effective, and experiments show that our method can improve the Oriented regions with convolutional neural networks features (R-CNN) by about 1.45 mAP on the FAIR1M benchmark.


## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.

## Get Started

Please refer to [oriented_model_starting.md](docs/oriented_model_starting.md) for training and testing.

## Citation
This repo is based on [OBBDetection](https://github.com/jbwang1997/OBBDetection).

This is the official implement of [Fine-Grained Feature Enhancement](configs/obb/oriented_rcnn). if it is used in your research, please cite the following information.

```
@article{zhou2022fine,
  title={Fine-Grained Feature Enhancement for Object Detection in Remote Sensing Images},
  author={Zhou, Yong and Wang, Sifan and Zhao, Jiaqi and Zhu, Hancheng and Yao, Rui},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={19},
  pages={1--5},
  year={2022},
  publisher={IEEE}
}
```
