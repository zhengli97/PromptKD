# PromptSRC Training

We provide bash scripts in [scripts/](../scripts) for training PromptSRC and independent V-L prompting baseline.
Make sure to update the `DATA` variable with dataset path in the script file and run the commands from the main directory `PromptSRC/`.
Below we provide training and testing instructions for PromptSRC. The same instructions are applicable for the baseline *independent V-L prompting* approach, MaPLe, CoOp and CoCoOp.

### Training time and compute
We train PromptSRC on each dataset with a batch size of 4 using a **single** NVIDIA A100 GPU.
Training PromptSRC on ImageNet for 20 epochs takes around 6 hours for a single seed. So results for 3 seeds takes around 18 hours. For all remaining 10 datasets, it combinedly takes around around 8 hours (for all 3 seeds) on a single A100 GPU. 

## PromptSRC

#### (1) Base-to-Novel class generalization setting
The base-to-novel PromptSRC configuration is provided in config file at `configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml`. All hyper-parameters such as GPA STD, GPA Mean, SCL loss weights coefficients, prompt length and prompt depth etc., can be modified using this config file.

Run the commands below to train PromptSRC on ImageNet.

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# seed=1
# trains and evaluates on base classes
bash scripts/promptsrc/base2new_train.sh imagenet 1
# evaluates on novel classes
bash scripts/promptsrc/base2new_test.sh imagenet 1

# seed=2
# trains and evaluates on base classes
bash scripts/promptsrc/base2new_train.sh imagenet 2
# evaluates on novel classes
bash scripts/promptsrc/base2new_test.sh imagenet 2

# seed=3
# trains and evaluates on base classes
bash scripts/promptsrc/base2new_train.sh imagenet 3
# evaluates on novel classes
bash scripts/promptsrc/base2new_test.sh imagenet 3
```

#### Averaging results over 3 seeds: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– PromptSRC/
|   |   |   |   |   |–– vit_b16_c2_ep20_batch4_4+4ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– PromptSRC/
|   |   |   |   |   |–– vit_b16_c2_ep20_batch4_4+4ctx/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Now use the script `parse_test_res.py` and run the commands below to calculate the averaged results:
```bash
# prints averaged results for base classes
python output/base2new/train_base/imagenet/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx --test-log
# averaged results for novel classes
python output/base2new/test_new/imagenet/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx --test-log
```

The above steps can be repeated for other individual datasets.

#### (2) Cross-Dataset Transfer setting
We provide instructions to train PromptSRC on ImageNet using all 1000 classes with 16 shots and then evaluating it directly on new downstream datasets.
The corresponding cross-dataset config for PromptSRC is available at: `configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets.yaml`. All PromptSRC hyper-parameters can be modified in this config file.
* Firstly, train PromptSRC on imagenet in few-shot manner (for all 3 seeds).

```bash
# seed=1 
bash scripts/promptsrc/xd_train.sh imagenet 1
# seed=2 
bash scripts/promptsrc/xd_train.sh imagenet 2
# seed=3 
bash scripts/promptsrc/xd_train.sh imagenet 3
```

* Now directly evaluate the ImageNet trained model on downstream cross-datasets.

```bash
# Other possible dataset values includes [imagenet, food101, dtd, ucf101, oxford_flowers, fgvc_aircraft, sun397, eurosat]

for SEED in 1 2 3
do
    bash scripts/promptsrc/xd_test.sh caltech101 ${SEED}
    bash scripts/promptsrc/xd_test.sh oxford_pets ${SEED}
    bash scripts/promptsrc/xd_test.sh stanford_cars ${SEED}
done
```
You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.


#### (3) Domain Generalization setting
We use the same ImageNet trained PromptSRC model for domain generalization experiments. The steps are similar to above cross-dataset experiments, however, the trained model is now evaluated on ImageNet variants.
The corresponding domain generalization config for PromptSRC is available at: `configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets.yaml`.
* Evaluate ImageNet model on different variants of ImageNet (datasets with domain shifts).

```bash
for SEED in 1 2 3
do
    bash scripts/promptsrc/xd_test.sh imagenetv2 ${SEED}
    bash scripts/promptsrc/xd_test.sh imagenet_sketch ${SEED}
    bash scripts/promptsrc/xd_test.sh imagenet_a ${SEED}
    bash scripts/promptsrc/xd_test.sh imagenet_r ${SEED}
done
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.

#### (4) Few-shot setting 
In this setting, PromptSRC is trained on all classes individual datasets with different few-shot splits (K = 1, 2, 4, 8, 16). The corresponding few-shot setting config for PromptSRC is available at: `configs/trainers/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot.yaml`.

Now use the training script `scripts/promptsrc/few_shot.sh` and run the commands below to calculate the results for imagenet dataset for all shots over 3 seeds:

```bash
# Other possible dataset values includes [caltech101, food101, dtd, ucf101, oxford_flowers, oxford_pets, fgvc_aircraft, stanford_cars, sun397, eurosat]

# train and test on given dataset for K=1 shot
bash scripts/promptsrc/few_shot.sh imagenet 1 
# train and test on given dataset for K=2 shot
bash scripts/promptsrc/few_shot.sh imagenet 2 
# train and test on given dataset for K=4 shot
bash scripts/promptsrc/few_shot.sh imagenet 4 
# train and test on given dataset for K=8 shot
bash scripts/promptsrc/few_shot.sh imagenet 8 
# train and test on given dataset for K=17 shot
bash scripts/promptsrc/few_shot.sh imagenet 16
```


You can obtain averaged results by using the script `parse_test_res.py` and following the similar steps as provided in base-to-novel generalization experiments.
<br>


#### Training and testing independent V-L prompting baseline approach

For training independent V-L prompting baseline approach, we provide their corresponding configs and scripts as follows.

```
configs
|–– datasets/
|–– trainers/
|   |–– CoCoOp/
|   |–– CoOp/
|   |–– IVLP/
|   |–– PromptSRC/
```

```
scripts
|–– cocoop/
|–– coop/
|–– promptsrc/
|–– independent-vlp/
```
    
Please use the corresponding config and script files and follow the same instructions as provided for PromptSRC for training and testing. 
This repository also supports using official [MaPLe](MaPLe.md), [CoOp](CoOp.md) and [Co-CoOp](Co-CoOp.md) configs and models.