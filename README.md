# VISAGE: Visual Ageing Generation

VISAGE is a project heavily inspired by [FRAN (Face Re-Aging Network)](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/), designed to modify the apparent age of faces in images and videos. It uses a U-Net architecture for frame-by-frame age transformation, providing a tool for visual effects in media without extensive manual editing. VISAGE aims to facilitate age alteration in visual content, supporting both aging and de-aging effects efficiently.

<a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/github/license/saltstack/salt" height=22.5></a>


## How does it work?
VISAGE is trained to take in an image of a face, an input age and an output age, and produce a 'delta' which is added to the original image to create the ageing effect. Unlike many GAN based methods of style (age) transfer, the face does not have to be encoded into a latent space representation, something which often distorts the identity of the subject. VISAGE is able to perform both **ageing and deageing**:

<p align="center">
<img src="docs/ageing.jpeg" width="1600px"/>
</p>

<p align="center">
<img src="docs/deageing.jpeg" width="1600px"/>
</p>

## Use in video
Because VISAGE inherently preserves the identity of the subject, we can use it for multiple different frames of a video sequence:


https://github.com/cassianlewis/VISAGE/assets/131266258/49451e60-7509-4404-81dc-ee345a4c1c04



The caveat here is that it doesn't work *perfectly*. There are still unwanted artefacts (although these should improve in future versions). 




## Table of Contents
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
  * [Pretrained Models](#pretrained-models)
  * [Training](#training)
    + [Preparing your Data](#preparing-your-data)
    + [Training VISAGE](#training-visage)
    + [Additional Notes](#additional-notes)
  * [Inference](#inference)
  * [Limitations](#limitations)
    + [Data limitations](#data-limitations)
    + [Other limitations](#other-limitations)
  * [Credits](credits)


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

### Additional notes
- The `augmentations` flag denotes whether to use random image augmentations (brightness, hue, rotation, crop etc) in the training data.


## Inference 
Having trained your model or if you're using a pretrained one, you can use `scripts/inference.py` to run inference on an image/set of images.
For example:
```
python scripts/inference.py \
--model_path saved_models/generator.pt
--data_path path/to/data
--input_age 30
--output_age 60
--output path/to/output/folder
```

This will process a folder of images, ageing the subject to the age specified in `output_age`.

The `show_delta` flag can be used to output the images like:

![0024](https://github.com/cassianlewis/VISAGE/assets/131266258/886c88c1-9b10-4c9a-ba65-18b06f84fcdd)

Without this flag, only the output image will be saved.


## Limitations
### Data limitations
A non-comprehensive list of limitations pertaining to the training data:
- The model is primarily trained on frontal-facing portraits ([FFHQ](https://github.com/NVlabs/ffhq-dataset) style). As such, it will underperform on side-shots.
- In a similar vein, it will not work well under certain lighting conditions (especially darker, cinematic shots).
- The background/clothing is often slightly altered by [SAMs](https://github.com/yuval-alaluf/SAM?tab=readme-ov-file) ageing shift. Subsequently, VISAGE may sometimes replicate this (although this can probably be ameliorated by masking the training data and/or output).
- The de-ageing process looks fairly airbrushed (again this is an artefact of the training data).
- The ageing process is restricted to the face (doesn't take into account changes to hair). 

### Other limitations
- The model is currently trained using a fixed aspect ration (1:1) and tensor input size (512 x 512 pixels). This is fairly rigid, and makes inputting other aspect ratios impossible for now. Although we can hack this via facial detection and cropping, for which I will release some scripts soon, I would prefer to have a model which can take in any aspect ratio (I need to look into whether this is possible).
- 512 x 512 pixels is not particularly high quality (I will train a 1024 x 1024 model in the future!).


## Credits
**FRAN: Ideas and Implementation Details** \
https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/


**SAM: Training Data**  
https://github.com/yuval-alaluf/SAM?tab=readme-ov-file \
Copyright (c) 2021 Yuval Alaluf \
License (MIT) https://github.com/yuval-alaluf/SAM/blob/master/LICENSE \

 


