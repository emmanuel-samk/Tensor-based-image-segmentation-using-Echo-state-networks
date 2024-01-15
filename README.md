# Analysis of Tensor-Based Image Segmentation using Echo State Network (ESN)
This Python package was used to explore the potential of the standard ESN model with linear regression readout for achieving good image segmentation results when applied on *Extended Structral Tensor* (EST)-based feature set. The *Structural Tensor* (ST) of a image is a feature matrix which elements consist of the averaged values of the gradient components in a certain neighbourhood defined around every point in an image. EST is ST extended to account for color/intensity components. It was proposed by Jackowski et al.[] and used to train some classical classifiers, such as Support Vecdtor Machine (SVM) for image segmentation. In this paper, we test the effect of the reservoir activations of EST on the accuracy of image segmentation. We evaluate the influence of spectral radius and reservoir size on the result and compare the best accuracy with the accuracy attained with Support Vector Machine. These results appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

 
## Problem Statement
The goal of any image segmentation technique is to achieve a high segmentation accuracy. In pixel classification based image segmentation, a clasifier is used to label every pixel in an image, such that pixels with similar features such as color, texture, are assigned the same label. The accuracy of such a technique is influenced by both the quality of the feature set used to train the classifier and the classification technique employed. The EST of color images conveys other information apart from color and intensity, such texture hidden in the neighborhood of pixels, thereby increasing segmentation accuracy []. Furthermore, feeding the pixel features into an ESN and training a classifier with the collected reservoir activations leads to increased segmentation accuracy[]. To the best of our knowledge, the accuracy of an image segmentation classifier trained with the collected reservoir activations of EST is however not known.

## Motivation
The main motivation for this tensor-based segmentation using ESN is to take advantage of the linear separability of ESN in addition to discriminative features defined by EST to achieve a high segmentation accuracy.

## Architecture of the Standard ESN Model
![standard ESN model](/docs/images/ESN.png)

## How it works
The EST-based features is fed into the ESN. Then, the reservoir output is used to train a linear regression classifier.

## Data Description
Instead of testing the accuracy on each of the images from which the training set was sampled as was done in [], we took the approach descripted in [] because of unavailability of 

The image datasets employed in this study were selected from the Berkley Segmentation Benchmark images published on *Berkley Computer Vision Group*'s website. The images are in JPG format, and each is of size (481 $\times$ 321), i.e., 154401 pixels. They are identified as '35058', '41033', '66053', '69040', '134052', '161062', and '326038', and stored in ![data/raw] (/data/raw).

Each dataset comprises 15 features and a class label indicating whether the pixel belongs to an object or a background. 
For each pixel, the values of the feature vector called *Extended Structural Tensor* (EST) convey information on image texture and curvature in the pixel's neighborhood and its intensity or color values. A detailed description of extracting features from an image using ST can be found in [].

### Training and Testing set
The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels." 
Since we could only obtain the training set, we combined them into one dataset and used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range, to lie between 0 and 1 using MinMax. 