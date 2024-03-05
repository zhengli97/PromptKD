# PromptKD: Unsupervised Prompt Distillation for Vision-Language Models 

<!-- This is the official PyToch implementation for "PromptKD: Unsupervised Prompt Distillation for Vision-Language Models." (CVPR 2024) -->


> [**PromptKD: Unsupervised Prompt Distillation for Vision-Language Models**]() <br>
> Zheng Li, Xiang Li*, Xinyi Fu, Xing Zhang, Weiqiang Wang, Shuo Chen, Jian Yang*. <br>
> CVPR 2024 <br>
> [[Paper]()] [[Project Page](https://zhengli97.github.io/PromptKD)] [[中文解读](https://zhengli97.github.io/PromptKD/chinese_interpertation.html)]


<!-- <hr />

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-regulating-prompts-foundational-model/prompt-engineering-on-imagenet)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet?p=self-regulating-prompts-foundational-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-regulating-prompts-foundational-model/prompt-engineering-on-imagenet-v2)](https://paperswithcode.com/sota/prompt-engineering-on-imagenet-v2?p=self-regulating-prompts-foundational-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-regulating-prompts-foundational-model/prompt-engineering-on-sun397)](https://paperswithcode.com/sota/prompt-engineering-on-sun397?p=self-regulating-prompts-foundational-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-regulating-prompts-foundational-model/prompt-engineering-on-ucf101)](https://paperswithcode.com/sota/prompt-engineering-on-ucf101?p=self-regulating-prompts-foundational-model)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-regulating-prompts-foundational-model/prompt-engineering-on-fgvc-aircraft-1)](https://paperswithcode.com/sota/prompt-engineering-on-fgvc-aircraft-1?p=self-regulating-prompts-foundational-model)

<hr /> -->


### Abstract

In this paper, we introduce an unsupervised domain prompt distillation framework, which aims to transfer the knowledge of a larger teacher model to a lightweight target model through prompt-driven imitation using unlabeled domain images. 

To our best knowledge, we are the first to (1) perform unsupervised domain-specific prompt-driven knowledge distillation for CLIP, and (2) establish a practical pre-storing mechanism of text features as shared class vectors between teacher and student.

### Framework

<div style="text-align:center"><img src="images/framework.png" width="100%" ></div>
<figcaption class="content has-text-left"  style="word-break:normal">Figure 1. An overview of our PromptKD framework. <strong>(a)</strong> We first pre-train a large CLIP teacher model with labeled training images. <strong>(b)</strong> Reuse the existing higher-quality teacher text features for unsupervised prompt distillation. <strong>(c)</strong> The well-trained student and pre-stored teacher text features are utilized for final inference.</figcaption>

### Highlights

(1). A novel two-stage unsupervised prompt distillation framework for Vision-Language Models.

(2). Reuse high-quality teacher text features instead of training the student's own text encoder.

(3). Distillation on large amounts of unlabeled domain images using soft labels provided by the teacher.

(4). PromptKD outperforms all existing prompt learning methods on 11 diverse recognition datasets.

## Experimental Results
Results reported below show accuracy for base and novel classes for across 11 recognition datasets averaged over 3 seeds.

### Base-to-Novel

<figure>
<img src="images/exp_results.png" alt="fail" width="100%"">
<figcaption class="content has-text-left" style="word-break:normal">Table 1. Comparison with existing state-of-the-art methods on base-to-novel generalization. Our PromptKD demonstrates strong generalization ability and achieves significant improvements on 11 recognition datasets given the <strong> ViT-B/16 image encoder</strong> of the CLIP model. The symbol △ denotes the performance improvement compared to the previous SOTA method.
</figure>

<figure>
<img src="images/hm_score.png" alt="fail" width="50%"">
<figcaption class="content has-text-centered" style="word-break:normal">Figure 1. Harmonic mean (HM) comparison on base-to-novel generalization.
</figure>


### Cross Dataset

<figure>
<img src="images/exp_results2.png" alt="fail" width="100%"">
<figcaption class="content has-text-left" style="word-break:normal">
Table 2. Comparison of PromptKD with existing advanced approaches on cross-dataset benchmark evaluation. 
Based on our pipeline, we perform unsupervised prompt distillation using the unlabeled domain data respectively (i.e., the transductive setting). 
The source model is trained on ImageNet. "ZSL" denotes the setting type for Zero-Shot Learning.
</figure>

## Running

### Preliminary

1. Create the environment and install Dassl.pytorch library. Please follow the instructions detailed in [INSTALL.md](docs/INSTALL.md).

2. (1) Pre-train your own large teacher CLIP model or (2) use our publicly released pre-trained teacher ViT-L/14 CLIP models. (Highly Recommended)   
After obtaining the teacher model, unzip these files and place the model in the `./teacher_models` folder.   
Our pre-trained teacher models are available at [Baidu Yun] [[TeraBox](https://terabox.com/s/1X4mxJtSaR8W2lrK5bsrCkg)] [Google Cloud]   
(Note that due to cloud space limitations, we only provide some models in Google Cloud. Sorry.)   

3. Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

4. Prepare the dataset. Please follow the instructions detailed in [DATASETS.md](docs/DATASETS.md).

### Run the script




## Contact

For any questions, please contact me via email (zhengli97[at]mail.nankai.edu.cn).

## Citation

If you find our paper or repo is helpful for your research, please consider citing our paper and giving this repo a ⭐.

<!-- ```

``` -->

## Acknowledgements

Our code is based on [PromptSRC](https://github.com/muzairkhattak/PromptSRC), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), [Co-CoOp and CoOp](https://github.com/KaiyangZhou/CoOp) repository. We thank the authors for releasing their code.
