# Vision Transformer for Brain Cancer Detection
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)

This is an application of the Vision Transformer for brain tumor detection through the use of hyperspectral images with a limited number of bands (i.e. using the [SLIM Brain Database](https://slimbrain.citsem.upm.es/)).

:bulb: An introduction to the ViT can be found here: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)\
:bulb: Parts of this work were inspired from: [SpectralFormer: Rethinking Hyperspectral Image Classification with Transformers](https://doi.org/10.48550/arXiv.2107.02988)


**Table of Contents**
- [Vision Transformer for Brain Cancer Detection](#vision-transformer-for-brain-cancer-detection)
  - [Requirements](#requirements)
  - [Directories and files](#directories-and-files)
  - [Results](#results)

## Requirements
- Pytorch 2.3.1 (+ all the packages in the requirements.txt file)
- Cuda 12.4 for parallelization

## Directories and files
* **data_analysis/** - Contains some scripts and notebooks used to study the composition of the SLIM Brain Database.
* **experiments/** - There are the code used to evaluate three experiments concerning the intra- and inter- partients classification. Moreover, there are the models trained as result from each experiment (pth format).
* **vit/** - Contains the ViT model used and the script used to run the hyperparameter optimization.

## Results
:dart: All the results obtained from the tests are explained in the following paper: [Vision Transformer for Brain Tumor Detection Using Hyperspectral Images with Reduced Spectral Bands]().
