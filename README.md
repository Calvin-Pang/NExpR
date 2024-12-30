# NExpR

This repository contains the official implementation for NExpR introduced in the following paper:

[**NExpR: Neural Explicit Representation for Fast Arbitrary-scale Medical Image Super-resolution**](https://www.sciencedirect.com/science/article/pii/S0010482524014392)
<br>
[Kaifeng Pang](https://kfpang.com), [Kai Zhao](https://kaizhao.net/), [Alex Ling Yu Hung](https://web.cs.ucla.edu/~alexhung/), [Haoxin Zheng](https://labs.dgsom.ucla.edu/mrrl/sunglab/haoxin_zheng), [Ran Yan](https://mrrl.ucla.edu/hulab/ran_yan), [Kyunghyun Sung](http://kyungs.bol.ucla.edu/Site/Home.html)
<br>
Computers in Biology and Medicine 184 (2025): 109354.

### Environment

You can install the required packages with this command:

```
pip install requirements.py
```

### Data

We use three public datasets and one in-house data in our experiments. The three public datasets are: [PROSTATEx](https://www.cancerimagingarchive.net/collection/prostatex/), [fastMRI](https://fastmri.med.nyu.edu/) and [MSD](http://medicaldecathlon.com/). We organize each dataset into two directories: `train` and `test`, containing 3D volumes in the `npy` format that will be processed into 2D images. 

Please ensure that your dataset is pre-processed in the same manner.

### Train

Our code supports multi-GPU training, and the configuration files are provided in this [directory](https://github.com/Calvin-Pang/NExpR/tree/main/configs). **Make sure to update the dataset paths in these configuration files to match your own directories.**

Then, for example, to train NExpR on the PROSTATEx dataset using 4 GPUs, you can run the following command with the specified save directory `SAVE_DIR` and experiment name `SAVE_NAME`:

```
torchrun --nproc_per_node=4 train_ddp.py --config configs/train_prostatex_cor_256_lite.yaml --dir SAVE_DIR --name SAVE_NAME
```

Feel free to modify the configuration files to adapt them to your own requirements.



### Test

### Citation

If you find our work useful in your research, please consider citing it:

```
@article{pang2025nexpr,
  title={NExpR: Neural Explicit Representation for fast arbitrary-scale medical image super-resolution},
  author={Pang, Kaifeng and Zhao, Kai and Hung, Alex Ling Yu and Zheng, Haoxin and Yan, Ran and Sung, Kyunghyun},
  journal={Computers in Biology and Medicine},
  volume={184},
  pages={109354},
  year={2025},
  publisher={Elsevier}
}
```
