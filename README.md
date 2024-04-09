# VISAGE: Visual Ageing Generation

VISAGE is a project inspired by [FRAN (Face Re-Aging Network)](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/), designed to modify the apparent age of faces in images and videos. It uses a U-Net architecture for frame-by-frame age transformation, providing a tool for visual effects in media without extensive manual editing. VISAGE aims to facilitate age alteration in visual content, supporting both aging and de-aging effects efficiently.


## Table of Contents
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
  * [Pretrained Models](#pretrained-models)
  * [Training](#training)
    + [Preparing your Data](#preparing-your-data)
    + [Training SAM](#training-sam)
    + [Additional Notes](#additional-notes)


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU is preferable, but inference will also run on your CPU (at a rate of approx 0.25 frames/s)
- Python 3

### Installation
- Dependencies:  
It is recommended to create a virtual environment to run this.
You can install the requirements by running `pip install requirements.txt`

## Pretrained Models
Please download the pretrained aging model from the following link.

| Path | Description
| :--- | :----------
|[VISAGE](https://drive.google.com/file/d/1zJsFYTAV5Oa-kEw71Id-Zx5UT-WxQjLn/view?usp=sharing)  | Trained VISAGE generator.

You can run this code to download it to the right place:

```
mkdir pretrained_models
pip install gdown
gdown "https://drive.google.com/file/d/1zJsFYTAV5Oa-kEw71Id-Zx5UT-WxQjLn/view?usp=sharing" --fuzzy -O pretrained_models/generator.pt
```

## Training
### Preparing your Data
Prepare your data according to the schema of `data/example`. Further instructions on how to download and/or create a dataset can be found in the `data` README.

### Training VISAGE
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `args.exp_dir`. This includes checkpoints and other training outputs (losses, intermediate images). 

If you want to train your own model, it is recommended to initialise the generator with the provided model via the `--model_path_gen` flag. This is because the pretrained model has already learnt useful representations which can aid/speed up the training of your own model. 


Training VISAGE can be achieved via this command:

```
python scripts/train.py \
--epochs=10 \
--batch_size=4 \
--model_path_gen=path/to/generator \
--data_path=path/to/data \
--exp_dir=experiments \
```

