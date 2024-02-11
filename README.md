# Tensor-Based Image Segmentation using Echo State Network (ESN)

## A pixel classification model for color image segmentation algorithm based on ESN and _Extended Structural Tensor_ (EST)-based pixel feature set

This package allows to create a [standard ESN](http://www.scholarpedia.org/article/Echo_state_network) model to classify the pixels in a color image (pixel-based image segmentation)
into two ground-truth regions, namely, object and background. The classification uses pixel features extracted by EST (see [] for details of how to compute the EST of an image). The primary motivation for this model is to take advantage of the discriminative features conveyed by EST-based features, such as texture hidden in the neighborhood of pixels, and the improved linear separability of inputs offered by the ESN reservoir to achieve high image segmentation accuracy.

To identify the (_spectral radius _ , _reservoir size_) pair that yields the best ESN classifier, it accepts a range of each of these parameters as input. Then, it performs a grid search of all possible (spectral radius , reservoir size) pairs and outputs the model with the best accuracy. The best accuracy can then be compared with the accuracy attained with other models such as Support Vector Machine. The results of the study for which this package was built appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

## Data Description

We use Berkley Segmentation Benchmark images to train and test the proposed model's performance. They are identified as '35058', '41033', '66053', '69040', '134052', '161062', and '326038' (see Table 1). Each is of size (481 $\times$ 321), i.e., 154401 pixels, and comes with ground-truth segmentations.

The datasets stored in ![data/raw] (/data/raw) are the EST the benchmark images extracted by the authors in [] and used to propose a similar pixel classification-based color image segmentation algorithm. Each dataset comprises 15 features and a class label indicating whether the pixel belongs to an object or a background. For each pixel, the feature vector conveys information on image texture and curvature in the pixel's neighborhood, its intensity or color values, and mix product of these.

### Training and Testing set

The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels."
Since we could only obtain the training set, we combined them into one dataset and used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range, to lie between 0 and 1 using MinMax.

The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels."
Since we could only obtain the training set, we used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range,
to lie between 0 and 1 using MinMax as follows.

## How it works

The EST-based pixel features $D$ are fed into the ESN. The reservoir $W$, acting as a high dimensional feature map, maps the inputs into a feature space $x$. The set $X$ of reservoir representations of the pixel features is then used as a new feature set to train the readout of the ESN - linear regression classifier.

![standard ESN model](/docs/images/ESN.png)

## References
