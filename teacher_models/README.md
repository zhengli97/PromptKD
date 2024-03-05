Put the pre-trained teacher CLIP-L/14 CLIP models in this folder.

The file structure should look like:
```
# for base-to-novel experiments:
$teacher_models/
|–– Caltech101/
    –– VLPromptLearner/
        –– model-best.pth.tar
|–– DescribableTextures/
|–– EuroSAT/
|–– FGVCaircraft/
|–– Food101/
|–– ImageNet/
|–– OxfordFlowers/
|–– OxfordPets/
|–– StandfordCars/
|–– SUN397/
|–– UCF101/

# for cross-dataset experiments:
|–– ImageNet-xd/
    –– VLPromptLearner_large/
        –– model.pth.tar-20
```