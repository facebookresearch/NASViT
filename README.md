# Summary

This repo is the official implementation of ICLR2022 paper ["NASVIT"](https://openreview.net/pdf?id=Qaw16njk6L). It currently includes code for training and the checkpoint.

# Pre-process

Before training, we move five images per class in the training dataset, and make these as the validation set. 

# Training

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=1024  main.py --cfg configs/cfg.yaml --amp-opt-level O0 --accumulation-steps 1`

# checkpoint 

[Download](https://drive.google.com/file/d/1Dk2yR7zHYB4dOiqCUnKjkCsKf_cMWjSY/view?usp=sharing)

**ImageNet Accuracy (val)**
| Model | Accuracy top-1 | Accuracy top-5 |
| :---: | :---: | :---: | 
| Smallest | 78.29 | 93.45 |
| Largest | 82.80 |  |
