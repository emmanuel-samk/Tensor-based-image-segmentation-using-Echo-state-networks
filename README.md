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
The datasets employed in this study were based on seven Berkley Segmentation Benchmark images published on *Berkley Computer Vision Group*'s website. 
The pictures are in JPG format, and each is of size (481 \by 321), i.e., 154401 pixels. They are identified as '35058',
'41033', '66053', '69040', '134052', '161062', and '326038'. They were used in a similar study to evaluate a tensor-based image segmentation algorithm using other classifiers []. 
The dataset for each image comprises 15 features and a class label indicating whether the pixel belongs to an object or a background. For each pixel, the values of the 
feature vector called *Extended Structural Tensor* (EST) conveys information on image texture and curvature in the neighborhood of the pixel and its intensity or color values.
A detailed description of extracting features from an image using EST can be found in [].

 The training set consists of uniformly distributed pixels selected from the pictures 
8068, 41033, 61060, 69040. Accuracy is estimated over entire images i.e., all of the pixels were classified and compared with their original labels.
