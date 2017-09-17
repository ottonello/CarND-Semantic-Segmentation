# Semantic Segmentation Using Fully Convolutional Networks
### Introduction
This project makes use of a Fully Convolutional Network (FCN) to perform
semantic segmentation on the image of a road. The model will take any number of
classes, but the training mentioned here is done with two labels('road' and 'not road').


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Implementation
The model is based in a VGG-16 model which is used as the encoder part of the FCN. The 
fully connected layers are first replaced by one-by-one convolutions in the model, which allows preserving the spatial information.
The decoder part consists of three convolutional layers, which progressively upsample up to the original image size. Upsampling is performed by using kernel and strides in a way that the resulting convolution is larger than the original. Skip layers are used so that different image block sizes have influence in the results.

Regularization is used to keep the weights at relatively small values, and dropout is used to prevent overfitting.

Below is the final model layers and dimensions:

%TODO IMAGE here

Training
=======
The model is trained on the Kitty Road Dataset with about 300 training images. This is not a lot of samples so dataset augmentation will surely help. Employed augmentation tecniques be shown later.

The training is performed with batch size of 8, as larger sizes would use up the available GPU memory.
After several tests it was found the network doesn't overfit

%TODO COMPLETE

Augmentation
=======
Dataset augmentation is performed by altering the brightness and contrast of the training images, as well as flipping them horizontally on a random basis. Later the result is limited to the [0,255] range. For better performance and implementation cleanness Tensorflow Image processing is used directly as part of the training pipeline. The augmentation is performed in the 'augment_op()' method.

Some augmentation samples:

% TODO add images 

Results
======

