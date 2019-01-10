# PyTorchCV: A PyTorch-Based Framework for Deep Learning in Computer Vision
```
@misc{CV2018,
  author =       {Donny You (youansheng@gmail.com)},
  howpublished = {\url{https://github.com/youansheng/PyTorchCV}},
  year =         {2018}
}
```

This repository provides source code for some deep learning based cv problems. We'll do our best to keep this repository up to date.  If you do find a problem about this repository, please raise it as an issue. We will fix it immediately.


## Implemented Papers

- [Image Classification](https://github.com/youansheng/PyTorchCV/tree/master/methods/cls)
    - VGG: Very Deep Convolutional Networks for Large-Scale Image Recognition
    - ResNet: Deep Residual Learning for Image Recognition
    - DenseNet: Densely Connected Convolutional Networks
    - MobileNetV2: Inverted Residuals and Linear Bottlenecks
    - ResNeXt: Aggregated Residual Transformations for Deep Neural Networks
    - SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
    - ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    - ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design

- [Semantic Segmentation](https://github.com/youansheng/PyTorchCV/tree/master/methods/seg)
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - PSPNet: Pyramid Scene Parsing Network
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
    
- [Object Detection](https://github.com/youansheng/PyTorchCV/tree/master/methods/det)
    - SSD: Single Shot MultiBox Detector
    - Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    - YOLOv3: An Incremental Improvement
    - FPN: Feature Pyramid Networks for Object Detection

- [Pose Estimation](https://github.com/youansheng/PyTorchCV/tree/master/methods/pose)
    - CPM: Convolutional Pose Machines
    - OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

- [Instance Segmentation](https://github.com/youansheng/PyTorchCV/tree/master/methods/seg)
    - Mask R-CNN


## Performances with PyTorchCV

#### Image Classification
- ResNet: Deep Residual Learning for Image Recognition

#### Semantic Segmentation
- PSPNet: Pyramid Scene Parsing Network

| Model | Backbone | Training data  | Testing data | mIOU | Pixel Acc | Setting |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|
| [PSPNet Origin](https://github.com/hszhao/PSPNet) | 3x3-ResNet101 | ADE20K train | ADE20K val | 41.96 | 80.64 | - |
| [PSPNet Ours](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [7x7-ResNet101](https://drive.google.com/open?id=1ROewKyaGPynox_-a50wHkSv1-0jYWyvc) | ADE20K train | ADE20K val | 44.18 | 80.91 | [PSPNet](https://github.com/youansheng/PyTorchCV/blob/master/hypes/seg/ade20k/fs_pspnet_ade20k_seg.json) |

#### Object Detection
- SSD: Single Shot MultiBox Detector

| Model | Backbone | Training data  | Testing data | mAP | FPS  | Setting |
|--------|:-------:|:---------:|:------:|:------:|:------:|:------:|
| [SSD-300 Origin](https://github.com/weiliu89/caffe/tree/ssd) | VGG16 | VOC07+12 trainval | VOC07 test | 0.772 | - | - |
| [SSD-300 Ours](https://drive.google.com/open?id=15J5blVyZq7lqCePh-Q8S2pxim3-f_8LP) | [VGG16](https://drive.google.com/open?id=1nM0UwmqR4lIHzmRWvs71jfP_gAekjuKy) | VOC07+12 trainval | VOC07 test | 0.786 | - | [SSD300](https://github.com/youansheng/PyTorchCV/blob/master/hypes/det/voc/ssd_vgg300_voc_det.json) |
| [SSD-512 Origin](https://github.com/weiliu89/caffe/tree/ssd) | VGG16 | VOC07+12 trainval | VOC07 test | 0.798 | - | - |
| [SSD-512 Ours](https://drive.google.com/open?id=1RF5gnqfiyz-EcSFU1OSK7tNuX_VRObVW) | [VGG16](https://drive.google.com/open?id=1nM0UwmqR4lIHzmRWvs71jfP_gAekjuKy) | VOC07+12 trainval | VOC07 test | 0.808 | - | [SSD512](https://github.com/youansheng/PyTorchCV/blob/master/hypes/det/voc/ssd_vgg512_voc_det.json) |

- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

| Model | Backbone |  Training data  | Testing data | mAP | FPS  | Setting |
|--------|:--------:|:---------:|:------:|:------:|:------:|:------:|
| [Faster R-CNN Origin](https://github.com/rbgirshick/py-faster-rcnn) | VGG16 | VOC07 trainval | VOC07 test | 0.699 | - | - |
| [Faster R-CNN Ours](https://drive.google.com/open?id=15SfklRiI1McVWEq9EAceznK-9sxXSQR4) | [VGG16](https://drive.google.com/open?id=1ZL9SS9KRzsDQhMe8kyPQ1LHA60wx_Vcj) | VOC07 trainval | VOC07 test | 0.706 | - | [Faster R-CNN](https://github.com/youansheng/PyTorchCV/blob/master/hypes/det/voc/fr_vgg16_voc_det.json) |

- YOLOv3: An Incremental Improvement

#### Pose Estimation
- OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

#### Instance Segmentation
- Mask R-CNN

## Commands with PyTorchCV

Take OpenPose as an example.
- Train the openpose model
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --base_lr 0.001 \
                --phase train \
                --gpu 0 1
```

- Finetune the openpose model
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --base_lr 0.001 \
                --phase train \
                --resume checkpoints/pose/coco/coco_open_pose_65000.pth \
                --gpu 0 1
```

- Test the openpose model(test_img):
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --phase test \
                --resume checkpoints/pose/coco/coco_open_pose_65000.pth \
                --test_img val/samples/ski.jpg \
                --gpu 0
```

- Test the openpose model(test_dir):
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --phase test \
                --resume checkpoints/pose/coco/coco_open_pose_65000.pth \
                --test_dir val/samples \
                --gpu 0
```

## Examples with PyTorchCV

<div align="center">

<img src="val/examples/pose/coco/000000319721_vis.png" width="500px"/>

<p> Example output of <b>VGG19-OpenPose</b></p>

<img src="val/examples/pose/coco/000000475191_vis.png" width="500px"/>

<p> Example output of <b>VGG19-OpenPose</b></p>

</div>

