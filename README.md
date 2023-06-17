# GRES: Generalized Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gres-generalized-referring-expression-1/generalized-referring-expression-segmentation)](https://paperswithcode.com/sota/generalized-referring-expression-segmentation?p=gres-generalized-referring-expression-1)

**[🏠[Project page]](https://henghuiding.github.io/GRES/)** &emsp; **[📄[arXiv]](https://arxiv.org/abs/2306.00968)**  &emsp; **[📄[PDF]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.pdf)** &emsp; **[🔥[New Dataset Download]](https://github.com/henghuiding/gRefCOCO)**

This repository contains code for **CVPR2023** paper:
> [GRES: Generalized Referring Expression Segmentation](https://arxiv.org/abs/2306.00968)  
> Chang Liu, Henghui Ding, Xudong Jiang  
> CVPR 2023 Highlight, Acceptance Rate 2.5%


<div align="center">
  <img src="https://github.com/henghuiding/ReLA/blob/main/imgs/fig1.png?raw=true" width="100%" height="100%"/>
</div><br/>

## Installation:

The code is tested under CUDA 11.8, Pytorch 1.11.0 and Detectron2 0.6.

1. Install [Detectron2](https://github.com/facebookresearch/detectron2) following the [manual](https://detectron2.readthedocs.io/en/latest/)
2. Run `sh make.sh` under `gres_model/modeling/pixel_decoder/ops`
3. Install other required packages: `pip -r requirements.txt`
4. Prepare the dataset following `datasets/DATASET.md`

## Inference

```
python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]
```

## Training

Firstly, download the backbone weights (`swin_base_patch4_window12_384_22k.pkl`) and convert it into detectron2 format using the script:

```
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
python tools/convert-pretrained-swin-model-to-d2.py swin_base_patch4_window12_384_22k.pth swin_base_patch4_window12_384_22k.pkl
```

Then start training:
```
python train_net.py \
    --config-file configs/referring_swin_base.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_weights]
```

Note: You can add your own configurations subsequently to the training command for customized options. For example:

```
SOLVER.IMS_PER_BATCH 48 
SOLVER.BASE_LR 0.00001 
```

For the full list of base configs, see `configs/referring_R50.yaml` and `configs/Base-COCO-InstanceSegmentation.yaml`


## Models

Update: We have added supports for ResNet-50 and Swin-Tiny backbones! Feel free to use and report these resource-friendly models in your work.

| Backbone | cIoU | gIoU |
|---|---|---|
| Resnet-50 | 39.53 | 38.62 |
| Swin-Tiny | 52.26 | 54.44 |
| Swin-Base | 62.42 | 63.60 |

All models can be downloaded from:

[Onedrive](https://entuedu-my.sharepoint.com/:f:/g/personal/liuc0058_e_ntu_edu_sg/EqyL6nftLjdIihQG2rYirPoBk9G5QHGPiJZX_z62axS3ZQ?e=ahrche)

## Acknowledgement

This project is based on [refer](https://github.com/lichengunc/refer), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Detectron2](https://github.com/facebookresearch/detectron2), [VLT](https://github.com/henghuiding/Vision-Language-Transformer). Many thanks to the authors for their great works!

## BibTeX
Please consider to cite GRES if it helps your research.

```latex
@inproceedings{GRES,
  title={{GRES}: Generalized Referring Expression Segmentation},
  author={Liu, Chang and Ding, Henghui and Jiang, Xudong},
  booktitle={CVPR},
  year={2023}
}
```
