# Introduction
Image segmentation aims to split an image into regions corresponding to specific objects or areas in the image. Image segmentation can be considered as a pixel classification task.
In this approach, the low-level features extracted from an image are fed to a supervised learning model, and the model is  trained to assign labels to pixels such that the same 
label is given to pixels with similar features. Thus, the accuracy of such an approach depends on the quality of the features and the type of model used.
Structural Tensor was proposed as a feature extraction technique to take advantage of the texture information hidden between neighboring pixels. 

This Python package was used to investigate the performance of the standard Echo State Network(ESN) model for pixel-based image segmentation. The model
was trained with pixel features extracted with Structural Tensors (see Section 2 for a detailed description of the data).  
Some of the results of this study appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

## Architecture of the Standard ESN Model

![standard ESN model](/docs/images/ESN.png)

## Data Description
The image datasets employed in this study were selected from the Berkley Segmentation Benchmark images published on *Berkley Computer Vision Group*'s website.
The images are in JPG format, and each is of size (481 $\times$ 321), i.e., 154401 pixels. A sample of these images identified as '35058', '41033', '66053', '69040', '134052', '161062', and '326038',
and their corresponding ground-truth segmentations are shown below.

The datasets based on these images were extracted by Jackowski et al.[] using *structural tensor* and used for a similar study that proposes a pixel classification-based
color image segmentation algorithm. Each dataset comprises 15 features and a class label indicating whether the pixel belongs to an object or a background. 
For each pixel, the values of the feature vector called *Extended Structural Tensor* (EST) convey information on image texture and curvature in the pixel's neighborhood and its intensity or color values.
A detailed description of extracting features from an image using ST can be found in [].

### Training and Testing set
The training set consists of uniformly distributed pixels selected from the image datasets (8068, 41033, 61060, 69040), and the test set consists of all the pixels in each image.
The model's accuracy was tested for each image: all the pixels from the test image are "classified and compared with their original labels." 
The feature vector values are normalized between 0 and 1
