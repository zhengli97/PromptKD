# Installation

### Acknowledgement: This readme file for installing datasets is modified from [MaPLe's](https://github.com/muzairkhattak/multimodal-prompt-learning) official repository.

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n promptkd python=3.8

# Activate the environment
conda activate promptkd

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Clone PromptKD code repository and install requirements
```bash
# Clone PromptSRC code base
git clone https://github.com/zhengli97/PromptKD.git

cd PromptKD/
# Install requirements

pip install -r requirements.txt

cd ..
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
# original source: https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
