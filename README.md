# PromptKD: Unsupervised Prompt Distillationi for Vision-Language Models 

This is the official PyToch implementation for "PromptKD: Unsupervised Prompt Distillationi for Vision-Language Models." (CVPR 2024)

[[Paper]()] [[Project Page](https://zhengli97.github.io/PromptKD)] [[中文解读](https://zhengli97.github.io/PromptKD/chinese_interpertation.html)]

### Abstract

In this paper, we introduce an unsupervised domain prompt distillation framework, which aims to transfer the knowledge of a larger teacher model to a lightweight target model through prompt-driven imitation using unlabeled domain images. 

To our best knowledge, we are the first to (1) perform unsupervised domain-specific prompt-driven knowledge distillation for CLIP, and (2) establish a practical pre-storing mechanism of text features as shared class vectors between teacher and student.

### Framework

<div style="text-align:center"><img src="images/framework.png" width="100%" ></div>
<figcaption class="content has-text-left"  style="word-break:normal">Figure 2. An overview of our PromptKD framework. <strong>(a)</strong> We first pre-train a large CLIP teacher model with labeled training images. <strong>(b)</strong> Reuse the existing higher-quality teacher text features for unsupervised prompt distillation. <strong>(c)</strong> The well-trained student and pre-stored teacher text features are utilized for final inference.</figcaption>

### Highlights

(1). A novel two-stage unsupervised prompt distillation framework for Vision-Language Models.

(2). Reuse high-quality teacher text features instead of training the student's own text encoder.

(3). Distillation on large amounts of unlabeled domain images using soft labels provided by teacher.

(4). PromptKD outperforms all existing prompt learning methods on 11 diverse recognition datasets.


## Results
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

### Base-to-Novel Experiments

<figure>
<img src="images/exp_results.png" alt="fail" width="100%"">
<figcaption class="content has-text-left" style="word-break:normal">Table 1. Comparison with existing state-of-the-art methods on base-to-novel generalization. Our PromptKD demonstrates strong generalization ability and achieves significant improvements on 11 recognition datasets given the <strong> ViT-B/16 image encoder</strong> of the CLIP model. The symbol △ denotes the performance improvement compared to the previous SOTA method.
</figure>

<figure>
<img src="images/hm_score.png" alt="fail" width="50%"">
<figcaption class="content has-text-centered" style="word-break:normal">Figure 1. Harmonic mean (HM) comparison on base-to-novel generalization.
</figure>


### Cross Dataset Experiments

<figure>
<img src="images/exp_results2.png" alt="fail" width="100%"">
<figcaption class="content has-text-left" style="word-break:normal">
Table 2. Comparison of PromptKD with existing advanced approaches on cross-dataset benchmark evaluation. 
Based on our pipeline, we perform unsupervised prompt distillation using the unlabeled domain data respectively (i.e., the transductive setting). 
The source model is trained on ImageNet. "ZSL" denotes the setting type for Zero-Shot Learning.
</figure>

## Model Zoo





## Runing

### Requirements









<!-- ## Citation

If is repo is helpful for your research, please consider citing our paper and giving this repo a star. -->



## Contact

For any questions, please contact me via email (zhengli97@mail.nankai.edu.cn)

## Acknowledgements

Our code is based on PromptSRC, [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code.
