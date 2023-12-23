# Introduction
Image segmentation aims to split an image into regions corresponding to specific objects or areas in the image. Image segmentation can be considered as a pixel classification task.
In this approach, the low-level features extracted from an image are fed to a supervised learning model, and the model is  trained to assign labels to pixels such that the same 
label is given to pixels with similar features. Thus, the accuracy of such an approach depends on the quality of the features and the type of model used.
Structural Tensor was proposed as a feature extraction technique to take advantage of the texture information hidden between neighboring pixels. 

This Python package was used to investigate the performance of the standard Echo State Network(ESN) model for pixel-based image segmentation. The model
was trained with pixel features extracted with Structural Tensors (see Section 2 for a detailed description of the data).  
Some of the results of this study appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

The *Echo state network* (ESN) model provides an alternative approach to gradient-descent-based approaches for training Recurrent Neural Networks (RNNs). 
ESN consists of an RNN comprising fixed random weights called the *reservoir* and a supervised learning model called the *readout*. 
It demonstrates that RNNs can perform significantly well even when only a subset of the network weights are trained. When driven by
an input signal, the reservoir acts as a high-dimensional feature map that improves the linear separability of the input data. Furthermore, 
it preserves the nonlinear transformation of the input history in its internal states. The readout is trained as a l

ESN is known for 


Although it was proposed for temporal tasks, its performance for many non-temporal tasks has been studied. In [], the potential of the ESN reservoir 
to refine pixel features for color *image segmentation* and the influence of the above-mentioned parameters on the results has been investigated.
Since features based on RGB values alone do have poor discriminative power with regards to classification, Jackowski et al. []  proposed  *Structural tensor* 
as a feature extraction technique to exploit the texture and local curvature information hidden between neighboring pixels. Extended with intensity or color values, 
the resulting *extended structural tensor* also conveys information on image color. The authors studied the usefulness and effectiveness of different
classification algorithms when trained with features based on structural tensors and showed that this method can perform better than the classical method 
based on color and intensity attributes only.

In this study, we investigate an image segmentation that exploits the discriminative power of EST combined with the internal dynamics to refine pixel features.

Image segmentation aims to split an image into regions corresponding to specific objects or areas in the image. Image segmentation can be considered as a pixel classification task.
In this approach, the low-level features extracted from an image are fed to a supervised learning model, and the model is  trained to assign labels to pixels such that the same 
label is given to pixels with similar features. However, the accuracy of such an approach is impacted by several factors, including the quality of the features and the type of model used.

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
The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels." 
Since we could only obtain the training set, we used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range,
to lie between 0 and 1 using MinMax as follows. 
