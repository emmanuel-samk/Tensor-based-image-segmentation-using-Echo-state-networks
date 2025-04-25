# Tensor-based image segmentation using Echo State Network (ESN)

### A code for exploring the accuracy of pixel classification based on a standard ESN model and Extended Structural Tensor (EST) pixel feature set.

This package is based on scikit-learn modules. It is used to study the potential of the [standard ESN](http://www.scholarpedia.org/article/Echo_state_network) model for pixel classification when trained with EST feature set. See [[1]](https://link.springer.com/article/10.1007/s10044-015-0502-2) for details of how to compute the EST features of an image. The primary motivation for training an ESN model with EST-based feature set of color images is to achieve high image segmentation accuracy by taking advantage of the discriminative features conveyed by EST-based features, such as texture hidden in the neighborhood of pixels, and the improved linear separability of inputs offered by the ESN reservoir.

Becuase the performance of ESN is influenced by the reservoir size and spectral radius of the reservoir weight matrix, this package makes it possible to find the pair of these values that yields the best model accuracy. To this end, it accepts a range of each of these global parameters as input and performs a grid search of (spectral radius , reservoir size) pairs and outputs the model with the best accuracy.

It allows the use of one of either two readout structures: ridge regressino or a support vector machine (svm) classifier.

The objective is to attain a model to classify the pixels in a color image (pixel-based image segmentation) into two ground-truth regions, namely, object and background.

The results of the study for which this package was built appeared in the proceedings of [MESAS 2018](https://link.springer.com/chapter/10.1007/978-3-030-14984-0_36).

## Data Description

The experiment was done with Berkley Segmentation Benchmark images, which are identified as shown in Table 1. Each image is of size (481 $\times$ 321), i.e., 154401 pixels, and comes with ground-truth segmentations.

![image data](/docs/data_images.png)
Table 1

The datasets stored in [data/raw](/data/raw) are the EST of the images extracted by the authors in [[1]](https://link.springer.com/article/10.1007/s10044-015-0502-2) and used to propose a similar pixel classification-based color image segmentation algorithm. Each dataset comprises 15 features and a class label indicating whether the pixel belongs to an object or a background. For each pixel, the feature vector conveys information on image texture and curvature in the pixel's neighborhood, its intensity or color values, and mixed product of these.

### Normalization

### Training and Testing set

The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels."
Since we could only obtain the training set, we combined them into one dataset and used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range, to lie between 0 and 1 using MinMax.

The dataset for each image consists of uniformly distributed pixels selected from the EST for the image. In [], these were combined and used as a training set,
while the test set consisted of all the pixels in each image. Then, the model's accuracy was tested for each image: all the pixels from the test image were "classified and compared with their original labels."
Since we could only obtain the training set, we used 80% to train the ESN, and the remaining 20% was used to test it. But first, we normalized the tensor values of the pixels, which vary in range,
to lie between 0 and 1 using MinMax as follows.

## How it works

Given a pixel $S(t)$ at time $t$ with EST-based features $S_{1}(t),...S_{p}(t)$
The EST-based pixel features $D$ are fed into the ESN. The reservoir $W$, acting as a high dimensional feature map, maps the inputs into a feature space $x$. The set $X$ of reservoir representations of the pixel features is then used as a new feature set to train the readout of the ESN - linear regression classifier.

![standard ESN model](/docs/est_esn_diagram.png)
