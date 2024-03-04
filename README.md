# PromptKD: Unsupervised Prompt Distillationi for Vision-Language Models 

This is the official PyToch implementation for "PromptKD: Unsupervised Prompt Distillationi for Vision-Language Models." (CVPR 2024)

[[Paper]()] [[Project Page](https://zhengli97.github.io/PromptKD)] [[中文解读](https://zhengli97.github.io/PromptKD/chinese_interpertation.html)]

### Abstract

In this paper, we introduce an unsupervised domain prompt distillation framework, which aims to transfer the knowledge of a larger teacher model to a lightweight target model through prompt-driven imitation using unlabeled domain images. 

To our best knowledge, we are the first to (1) perform unsupervised domain-specific prompt-driven knowledge distillation for CLIP, and (2) establish a practical pre-storing mechanism of text features as shared class vectors between teacher and student.

### Framework


## Results
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

### Base-to-Novel Experiments

| Name                                       | Base Acc. | Novel Acc. |    HM     | 
|--------------------------------------------|:---------:|:----------:|:---------:|
| [CLIP](https://arxiv.org/abs/2103.00020)   |   69.34   |   74.22    |   71.70   | 
| [CoOp](https://arxiv.org/abs/2109.01134)   |   82.69   |   63.22    |   71.66   |
| [CoCoOp](https://arxiv.org/abs/2203.05557) |   80.47   |   71.69    |   75.83   |
| [ProDA](https://arxiv.org/abs/2205.03340)  |   81.56   |   75.83    |   76.65   | 
| [MaPLe](https://arxiv.org/abs/2210.03117)  |   82.28   |   75.14    |   78.55   |
| [PromptSRC](https://arxiv.org/abs/2307.06948)| 84.26   |   76.10    |   79.97   |
| [PrompKD]()                                  | 86.96   |   80.73    |   83.73   |

### Cross Dataset Experiments




## Model Zoo





## Runing

### Requirements









<!-- ## Citation

If is repo is helpful for your research, please consider citing our paper and giving this repo a star. -->



## Contact

For any questions, please contact me via email (zhengli97@mail.nankai.edu.cn)

## Acknowledgements

Our code is based on PromptSRC, [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code.
