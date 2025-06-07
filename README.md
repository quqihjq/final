## Class Prototype-Driven Global-Local Discriminative Weakly Supervised Semantic Segmentation

<div align="center">

<br>
  <img width="100%" alt="CPDGL flowchart" src="./docs/imgs/CPDGL.png">
</div>

<!-- ## Abastract -->

We proposed Token Contrast to address the over-smoothing issue and further leverage the virtue of ViT for the Weakly-Supervised Semantic Segmentation task.

## Data Preparations
<details>
<summary>
VOC dataset
</summary>

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar –xvf VOCtrainval_11-May-2012.tar
```
#### 2. Download the augmented annotations
The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012`. The directory sctructure should thus be 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```
</details>

<details>

<summary>
COCO dataset
</summary>

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```
#### 2. Generating VOC style segmentation labels for COCO
To generate VOC style segmentation labels for COCO dataset, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc). Or, just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

I recommend to organize the images and labels in `coco2014` and `SegmentationClass`, respectively.

``` bash
MSCOCO/
├── coco2014
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```



</details>

## Create environment

### Clone this repo

```bash
git clone https://github.com/quqihjq/final
```

### Train

## Citation
Please kindly cite our paper if you find it's helpful in your work.

``` bibtex
@inproceedings{ru2023token,
    title = {Token Contrast for Weakly-Supervised Semantic Segmentation},
    author = {Lixiang Ru and Heliang Zheng and Yibing Zhan and Bo Du}
    booktitle = {CVPR},
    year = {2023},
  }
```

## Acknowledgement

We mainly use [ViT-B](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vit.py) and [DeiT-B](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/deit.py) as the backbone, which are based on [timm](https://github.com/huggingface/pytorch-image-models). 
