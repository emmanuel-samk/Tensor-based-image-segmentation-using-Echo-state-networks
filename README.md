# Introduction
Image segmentation aims to split an image into regions corresponding to specific objects or areas in the image. Image segmentation can be considered as a pixel classification task.
In this approach, the low-level features extracted from an image are fed to a supervised learning model, and the model is  trained to assign labels to pixels such that the same 
label is given to pixels with similar features. Thus, the accuracy of such an approach depends on the quality of the features and the type of model used.
To take advantage of the information on texture, which is hidden between neighboring pixels, Structural Tensors was proposed as a feature extraction technique. 

This Python package was used to investigate the performance of the standard Echo State Network(ESN) model for pixel-based image segmentation. The model
was trained with pixel features extracted with Structural Tensors (see Section 2 for a detailed description of the data).  
Some of the results of this study appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

## Architecture of the Standard ESN Model

![standard ESN model](/docs/images/ESN.png)

## Data Description
This data was used to evaluate a Tensor-based Image Segmentation Algorithm (TBISA) proposed in [].
