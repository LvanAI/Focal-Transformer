# MindSpore Focal-Transformer

## Introduction

This work is used for reproduce Focal-Transformer based on NPU(Ascend 910)

**Focal-Transformer** is introduced in [arxiv](https://arxiv.org/abs/2107.00641)

Currently a good way to reduce the computational and memory cost and improve performance in vision transformer is to apply coarse-grained global attentions or fine-grained local attentions. However, bothapproaches cripple the modeling power of the original self-attention mechanism of multi-layer Transformers, thus leading to sub-optimal solutions. A new mechanism, called focal self-attention, incorporates both fine-grainedlocal and coarse-grained global interactions where each token attends its closest surrounding tokens at fine granularity and the tokens far awayat coarse granularity, and thus can capture both short- and long-range visual dependencies efficiently. 

Focal-Transformer achieves strong performance on ImageNet classification (83.6 on val with 9.1G flops)

![framework](/figures/focal-transformer-teaser.png)

## Data preparation

Download and extract [ImageNet](https://image-net.org/).

The directory structure is the standard layout for the MindSpore [`dataset.ImageFolderDataset`](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/dataset/mindspore.dataset.ImageFolderDataset.html?highlight=imagefolderdataset), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
## Training


```
mpirun -n 8 python train.py --config <config path> > train.log 2>&1 &
```

## Evaluation 


```
python eval.py --config <config path>
```


## Acknowledgement

We heavily borrow the code from [Focal-Transformer](https://github.com/microsoft/Focal-Transformer) and [swin_transformer](https://gitee.com/mindspore/models/tree/master/research/cv/swin_transformer)
We thank the authors for the nicely organized code!
