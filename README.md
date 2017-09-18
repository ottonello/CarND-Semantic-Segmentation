# Semantic Segmentation Using Fully Convolutional Networks
### Introduction
This project makes use of a Fully Convolutional Network (FCN) to perform
semantic segmentation on the image of a road. The model will take any number of
classes, but the training mentioned here is done with two labels('road' and 'not road').


### Setup
##### Frameworks and Packages
Make sure you have the following installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Implementation
The model is based in a VGG-16 model which is used as the encoder part of the FCN. The 
fully connected layers are first replaced by one-by-one convolutions in the model, which allows preserving the spatial information.
The decoder part consists of three convolutional layers, which progressively upsample up to the original image size. 
Upsampling is performed by using kernel and strides in a way that the resulting convolution is larger than the original.
Skip layers are used so that different image block sizes have influence in the results.

I attempted using regularization but the results weren't as good as without it, it would be worth investigating what happened around that.

Below is the final model with layers and dimensions, including the skipped stages:

![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/graph-run%3D%20(2).png)

Training
=======
The model is trained on the Kitty Road Dataset with about 300 training images. 
This is not a lot of samples so dataset augmentation will surely help get better results. Employed augmentation tecniques be shown later.

The training is performed with batch size of 8, as larger sizes would use up the available GPU memory. Training was done with
learning rate = 1e-4, as larger values would not converge. A dropout probability of 0.5 was used to avoid overfitting. 

Augmentation
=======
Dataset augmentation is performed by altering the brightness and contrast of the training images, as well as flipping them horizontally 
on a random basis. Later the result is limited to the [0,255] range. For better performance and implementation cleanness Tensorflow Image
 processing is used directly as part of the training pipeline. The augmentation is performed in the 'augment_op()' method.

Some augmentation samples:

![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/0.png)
![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/1.png)
![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/2.png)
![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/3.png)
![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/4.png)
![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/5.png)
![model](https://github.com/ottonello/CarND-Semantic-Segmentation/blob/master/results/augm/10.png)


Results
======

