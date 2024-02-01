# Tensor-Based Image Segmentation using [Echo State Network (ESN)](http://www.scholarpedia.org/article/Echo_state_network)

## A package that allows for the implementation of color image segmentation using an ESN classifier trained with _Extended Structural Tensor_ (EST)-based feature set.

It was used to investigate the potential of the standard ESN model with linear regression readout for achieving good image segmentation results when applied on _Extended Structral Tensor_ (EST)-based feature set.
The goal of any image segmentation technique is to achieve a high segmentation accuracy. In pixel classification based image segmentation, a clasifier is used to label every pixel in an image, such that pixels with similar features such as color, texture, are assigned the same label. The accuracy of such a technique is influenced by both the choice of a classifier and the quality of the feature set used to train the classifier. The _Structural Tensor_ (ST) of an image is a feature matrix in which elements consist of the averaged values of the gradient components in a specific neighborhood defined around every point in an image and EST is a ST extended to account for color/intensity components. Apart from color and intensity, the EST of an image conveys other discriminative feastures such as texture hidden in the neighborhood of pixels. It was proposed by Jackowski et al.[]. The authors showed that classifiers based on EST can outperform those based on color and intensity attributes only []. Furthermore, feeding pixel features into an ESN and training a classifier with the collected reservoir activations leads to increased segmentation accuracy, as the ESN reservoir improves the linear separability of the inputs []. Motivated by these two results, this package was built to:

- investigate the accuracy of color image segmentation based on ESN and EST-based feature set.
- evaluate the influence of the _spectral radius_ and _reservoir size_ of the ESN on accuracy
- compare the best accuracy with the accuracy attained with Support Vector Machine.
  These results appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

## How it works

The EST-based features is fed into the ESN. Then, the reservoir output is used to train a linear regression classifier.

![standard ESN model](/docs/images/ESN.png)

## Data Description

Instead of testing the accuracy on each of the images from which the training set was sampled as was done in [], we took the approach descripted in [] because of unavailability of

The image datasets employed in this study were selected from the Berkley Segmentation Benchmark images published on _Berkley Computer Vision Group_'s website. The images are in JPG format, and each is of size (481 $\times$ 321), i.e., 154401 pixels. They are identified as '35058', '41033', '66053', '69040', '134052', '161062', and '326038', and stored in ![data/raw] (/data/raw).

The image datasets employed in this study were selected from the Berkley Segmentation Benchmark images published on _Berkley Computer Vision Group_'s website.
The images are in JPG format, and each is of size (481 $\times$ 321), i.e., 154401 pixels. A sample of these images identified as '35058', '41033', '66053', '69040', '134052', '161062', and '326038',
and their corresponding ground-truth segmentations are shown below.

The datasets based on these images were extracted by Jackowski et al.[] using _structural tensor_ and used for a similar study that proposes a pixel classification-based
color image segmentation algorithm. Each dataset comprises 15 features and a class label indicating whether the pixel belongs to an object or a background.
For each pixel, the values of the feature vector called _Extended Structural Tensor_ (EST) convey information on image texture and curvature in the pixel's neighborhood and its intensity or color values.
A detailed description of extracting features from an image using ST can be found in [].

Each dataset comprises 15 features and a class label indicating whether the pixel belongs to an object or a background.
For each pixel, the values of the feature vector called _Extended Structural Tensor_ (EST) convey information on image texture and curvature in the pixel's neighborhood and its intensity or color values. A detailed description of extracting features from an image using ST can be found in [].

### Training and Testing set

The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels."
Since we could only obtain the training set, we combined them into one dataset and used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range, to lie between 0 and 1 using MinMax.

The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels."
Since we could only obtain the training set, we used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range,
to lie between 0 and 1 using MinMax as follows.
