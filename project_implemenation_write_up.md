# Hand gesture recognition using neural networks

## Developers

Venkateswarlu Jonnadula
Mohana silpa Palla

## Problem Statement

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

- Thumbs up:  Increase the volume
- Thumbs down: Decrease the volume
- Left swipe: 'Jump' backwards 10 seconds
- Right swipe: 'Jump' forward 10 seconds  
- Stop: Pause the movie

Each video is a sequence of 30 frames (or images)

## Understanding the Dataset

The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders.

## Two Architectures: 3D Convs and CNN-RNN Stack

After understanding and acquiring the dataset, the next step is to try out different architectures to solve this problem. 

For analysing videos using neural networks, two types of architectures are used commonly. 

One is the standard **CNN + RNN** architecture in which you pass the images of a video through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN. 

*Note:*
 - You can use transfer learning in the 2D CNN layer rather than training your own CNN 
 - GRU (Gated Recurrent Unit) or LSTM (Long Short Term Memory) can be used for the RNN

The other popular architecture used to process videos is a natural extension of CNNs - a **3D convolutional network**. In this project, we will try both these architectures.

## Data Preprocessing

We can apply several of the image procesing techniques for each of image in the frame.

### Resize

 We will convert each image of the train and test set into a matrix of size 120*120

### Cropping

Given that one of the data set is of rectangualr shape, we will crop that image to 120*120, this is different to resize, while resize changes the aspect ratio of rectangular image. In cropping we will center crop the image to retain the middle of the frame.

### Normalization

We will use mean normaliztion for each of the channel in the image.

## Data Agumentation

We have a total of 600+ for test set and 100 sampels for validation set. We will increase this 2 fold by usign a simple agumentiaton technique of affine transforamtion.

### Affine Transformation

In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from input image and their corresponding locations in output image. Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.

Check below example, and also look at the points I selected (which are marked in Green color):

``` python
img = cv2.imread('drawing.png')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

We will perform a same random affine transform for all the images in the frameset. This way we are generating new dataset from existing dataset.

### Flipping Images Horizontally

Note that fliiping images horizontally comes with special cavet, we need to swap the left swipe <-> right swipe as we flip the image.
This technique of image augmentation adds more generalization to the dataset.

## Generators

**Understanding Generators**: As you already know, in most deep learning projects you need to feed data to the model in batches. This is done using the concept of generators. 

Creating data generators is probably the most important part of building a training pipeline. Although libraries such as Keras provide builtin generator functionalities, they are often restricted in scope and you have to write your own generators from scratch. In this project we will implement our own cutom generator, our generator will feed batches of videos, not images. 

Let's take an example, assume we have 23 samples and we pick batch size as 10.

In this case there will be 2 complete batches of ten each
- Batch 1: 10
- Batch 2: 10
- Batch 3: 3

The final run will be for the remaining batch that was not part of the the full batch. 

Full batches are covered as part of the for loop the remainder are covered post the for loop.

Note: this also covers the case, where in batch size is day 30 and we have only 23 samples. In this case there will be only one single batch with 23 samples.

## Reading Video as Frames

Note that in our project, each gesture is a broken into indivdual frame. Each gesture consists of 30 individual frames. While loading this data via the generator there is need to sort the frames if we want to maintain the temporal information.

The order of the images loaded might be random and so it is necessary to apply sort on the list of files before reading each frame.


# Implementation 

## 3D Convolutional Network, or Conv3D

Now, lets implement a 3D convolutional Neural network on this dataset. To use 2D convolutions, we first convert every image into a 3D shape : width, height, channels. Channels represents the slices of Red, Green, and Blue layers. So it is set as 3. In the similar manner, we will convert the input dataset into 4D shape in order to use 3D convolution for : length, breadth, height, channel (r/g/b).

*Note:* even though the input images are rgb (3 channel), we will perform image processing on each frame and the end individual frame will be grayscale (1 channel) for some models

Lets create the model architecture. The architecture is described below:

While we tried with multiple ***filter size***, bigger filter size is resource intensive and we have done most experiment with 3*3 filter

We have used **Adam** optimizer with its default settings.
We have additionally used the ReduceLROnPlateau to reduce our learning alpha after 2 epoch on the result plateauing.


## Model #1

Build a 3D convolutional network, based loosely on C3D.

This model used 4 convolution layers each uses the filter size increasing in order [8,16,32,64]
Batch Normalisation applied on each convolution output.
Activation function used is 'relu'. Followed by Maxpooling layers.
Next steps are Flatten and Dense layers.

Below are parameters considered for Model1 
batch_size = 30
img_height = 128
img_width = 128
frames=30
channels=3
num_epochs = 20


Model Summary
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv3d_8 (Conv3D)           (None, 30, 128, 128, 8)   656       
                                                                 
 batch_normalization_6 (Batc  (None, 30, 128, 128, 8)  32        
 hNormalization)                                                 
                                                                 
 activation_8 (Activation)   (None, 30, 128, 128, 8)   0         
                                                                 
 max_pooling3d_8 (MaxPooling  (None, 15, 64, 64, 8)    0         
 3D)                                                             
                                                                 
 conv3d_9 (Conv3D)           (None, 15, 64, 64, 16)    3472      
                                                                 
 batch_normalization_7 (Batc  (None, 15, 64, 64, 16)   64        
 hNormalization)                                                 
                                                                 
 activation_9 (Activation)   (None, 15, 64, 64, 16)    0         
                                                                 
 max_pooling3d_9 (MaxPooling  (None, 7, 32, 32, 16)    0         
 3D)                                                             
                                                                 
 conv3d_10 (Conv3D)          (None, 7, 32, 32, 32)     4640      
                                                                 
 batch_normalization_8 (Batc  (None, 7, 32, 32, 32)    128       
 hNormalization)                                                 
                                                                 
 activation_10 (Activation)  (None, 7, 32, 32, 32)     0         
                                                                 
 max_pooling3d_10 (MaxPoolin  (None, 3, 16, 16, 32)    0         
 g3D)                                                            
                                                                 
 conv3d_11 (Conv3D)          (None, 3, 16, 16, 64)     18496     
                                                                 
 activation_11 (Activation)  (None, 3, 16, 16, 64)     0         
                                                                 
 dropout_6 (Dropout)         (None, 3, 16, 16, 64)     0         
                                                                 
 max_pooling3d_11 (MaxPoolin  (None, 1, 8, 8, 64)      0         
 g3D)                                                            
                                                                 
 flatten_2 (Flatten)         (None, 4096)              0         
                                                                 
 dense_6 (Dense)             (None, 256)               1048832   
                                                                 
 dropout_7 (Dropout)         (None, 256)               0         
                                                                 
 dense_7 (Dense)             (None, 128)               32896     
                                                                 
 dropout_8 (Dropout)         (None, 128)               0         
                                                                 
 dense_8 (Dense)             (None, 5)                 645       
                                                                 
=================================================================
Total params: 1,109,861
Trainable params: 1,109,749
Non-trainable params: 112
_________________________________________________________________

Model1 produces output of 
loss: 1.3165 - categorical_accuracy: 0.4928 - val_loss: 1.2108 - val_categorical_accuracy: 0.5000 

## Model #2

Build a 3D convolutional network, aka C3D.

Below are parameters considered for Model1 
batch_size = 20
img_height = 128
img_width = 128
frames=30
channels=3
num_epochs = 30

Model Summary

Model architecture in terms of filters sizes and number of layers, maxpooling layers hasn't changed,
so the summary doesn't change with number of parameters involved.

Since batch_size is reduced and number of epochs is increased, the learning rate is more and model has
pass thru the training data more times.  


Model2 produces output of 
loss: 1.2870 - categorical_accuracy: 0.5294 - val_loss: 1.2577 - val_categorical_accuracy: 0.5600 


## Model #3

Build a 3D convolutional network, aka C3D. In this model we used data Affine augmentation and flip augmentation,
with batch normalisation.   

Below are parameters considered for Model1 
batch_size = 20
img_height = 128
img_width = 128
frames=30
channels=3
num_epochs = 50

Model Summary

Model architecture in terms of filters sizes and number of layers, maxpooling layers hasn't changed,
so the summary doesn't change with number of parameters involved.


Since batch_size is reduced and number of epochs is increased, the learning rate is more and model has
pass thru the training data more times.  



Model3 produces output of 
loss: 0.9438 - categorical_accuracy: 0.7157 - val_loss: 0.8370 - val_categorical_accuracy: 0.6900 

We can cleary see, 
1. with lesser batch_size and more epochs 
2. With data augmentation techniques
the accuracy of model is improved a lot.

so we have taken Model3 as final model and same saved model is attached for submission.
