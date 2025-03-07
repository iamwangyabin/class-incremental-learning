# Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

### Papers

- Adaptive Aggregation Networks for Class-Incremental Learning,
CVPR 2021. \[[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Adaptive_Aggregation_Networks_for_Class-Incremental_Learning_CVPR_2021_paper.pdf)\] \[[Project Page](https://class-il.mpi-inf.mpg.de/)\]

- Mnemonics Training: Multi-Class Incremental Learning without Forgetting,
CVPR 2020. \[[PDF](https://arxiv.org/pdf/2002.10211.pdf)\] \[[Project Page](https://class-il.mpi-inf.mpg.de/mnemonics-training/)\]

### Citations

Please cite our papers if they are helpful to your work:

```bibtex
@inproceedings{Liu2020AANets,
  author    = {Liu, Yaoyao and Schiele, Bernt and Sun, Qianru},
  title     = {Adaptive Aggregation Networks for Class-Incremental Learning},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2544-2553},
  year      = {2021}
}
```

```bibtex
@inproceedings{liu2020mnemonics,
author    = {Liu, Yaoyao and Su, Yuting and Liu, An{-}An and Schiele, Bernt and Sun, Qianru},
title     = {Mnemonics Training: Multi-Class Incremental Learning without Forgetting},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
pages     = {12245--12254},
year      = {2020}
}
```

### Acknowledgements

Our implementation uses the source code from the following repositories:

* [Learning a Unified Classifier Incrementally via Rebalancing](https://github.com/hshustc/CVPR19_Incremental_Learning)

* [iCaRL: Incremental Classifier and Representation Learning](https://github.com/srebuffi/iCaRL)

* [Dataset Distillation](https://github.com/SsnL/dataset-distillation)

* [Generative Teaching Networks](https://github.com/uber-research/GTN)


AANet+lucir cifar 50-10
python main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --resume_fg --ckpt_dir_fg ./logs/cifar100_nfg50_ncls10_nproto20_lucir_dual_b1ss_b2free_fixed_exp01_aanetTPCIL_noloss13/iter_4_b1.pth --notes=aanet_lucir


aanet做的测试代码，base是lucir 是cifar100 50-10

python main_attacker.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --resume_fg --ckpt_dir_fg ./logs/cifar100_nfg50_ncls10_nproto20_lucir_dual_b1ss_b2free_fixed_exp01_aanetTPCIL_noloss13/iter_4_b1.pth --notes=tpcli_noloss23
python main_attacker.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --resume_fg --ckpt_dir_fg ./logs/cifar100_nfg50_ncls10_nproto20_lucir_dual_b1ss_b2free_fixed_exp01_aanetTPCIL_noloss13/iter_4_b1.pth --notes=aalucir_ak




python main_attacker.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --baseline=lucir --branch_mode=dual --branch_1=free --branch_2=free --fusion_lr=0.0 --dataset=cifar100 --resume_fg --ckpt_dir_fg ./logs/cifar100_nfg50_ncls10_nproto20_lucir_dual_b1ss_b2free_fixed_exp01_aanetTPCIL_noloss13/iter_4_b1.pth --notes=tpcil_dulebrach