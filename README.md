# Behavioral-cloning-for-a-self-driving-vehicle

The ultimate goal of this project is to build a deep learning network which predicts a self-driving vehicle's steering angle as a function of the view ahead of it, i.e. the images coming from its front center camera (with an option to use left and right hand side cameras, too). The steering angle should always be predicted in such a way that no tire leaves the drivable portion of the track surface; the car does not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). 

The key steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track as described above
* Summarize the results with a written report

Below is a detailed description of how I went about collecting the data, choosing an appropriate model architecture, training and tuning the model and documenting the work and results. This repository contains the following files:

* training\_model\_20.py - python script to create and train the model
* drive.py - script to test-drive the car through the simulator
* video.mp4 - video demonstrating the behavior of my vehicle as dictated by the predictions of the model 

The saved Keras model file, `model\_20.h5`, which contains the trained model architecture and weights is over 400MB in size and therefore, could not be uploaded to GitHub. I'm happy to share it upon request. 

## Data collection

I used the software provided by Udacity to simulate good driving under different environments, as represented by the images coming from car cameras. To create a balanced dataset, I "drove" 2 laps clockwise and 2 laps counter-clockwise. This way, we can avoid a bias which can be introduced in the dataset if one of the directions dominates in the steering angle. I also simulated car steering in abnormal situations, e.g. when the car is at or across the edge of the road, etc., with the purpose of "teaching" the car how to recover under such circumstances. I ended up with approximately 15,000 instances of normal driving and approximately 10,000 instances of recovery situations. Each instance consists of one steering angle measurement and three images collected from the center, left and right cameras.

## Data preparation 

I used only the center camera images for model training. Thus, the input feature is an image with dimensions (160, 320, 3). First of all, I shuffled the data. Then I scaled the pixel values around a mean of zero, in the range -0.5 to +0.5. Additionally, I cropped 70 pixels of the image height from the top and 20 pixels from the bottom, as the sky, trees, etc. and the portion of the vehicle hood at the bottom, probably do not serve as meanigful inputs for steering angle prediction. 

In initial experimentation, when I had a smaller dataset collected in one-directional driving only, I had also performed data augmentation by including flipped images and measurements. However, as my final dataset contains a large and rich data of both clockwise and counter-clockwise driving, as well as recovery situations, I considered it unnecessary to perform additional augmentation for the training of the final model. 

## Model architecture 

My final architecture is based on LeNet-5, which consists of two convolutional layers, each followed by rectified linear unit (relu) activation function and max pooling and three fully connected layers, also followed by relu activations. More detailed descrition is below:

* Input 160x320x3, as I'm using colored images
* Cropping of top 70 and 20 bottom pixels applied via Keras' Cropping2D method
* Convolutional layer with depth of 6 (6,5,5) followed by a Rectified Linear Unit (ReLu) activation
* Dropout layer with a keep probability of 0.25
* Another convolutional layer, with depth of 16 (16,5,5) followed by ReLu activation 
* Dropout layer with a keep probability of 0.25
* Fully connected layer of 120 nodes followed by ReLu activation
* Fully connected layer of 84 nodes followed by ReLu activation
* Final output value

I chose Mean Squared Error (MSE) as the model loss objective function, as we are dealing with a regression, as opposed to a classification. And I used Adam optimizer.  

I beleive the above architecture is appropriate, as (i) it is based on a well-known model, whose effectiveness is proven for image recognition; (ii) sufficient non-linearity and granularity has been designed in it through the use of ReLu layers; (iii) dropout layers have been applied to reduce overfitting. 

## Best practices applied to modeling 

To avoid overfitting, I included in the model Dropout layers with keep probability of 25%. Initially, I had used Max Pooling layers but it became clear that more granularity was required for reliable predictions. With trial-and-error, I concluded that no Max Pooling but two Dropout layers with keep probability of 25% represented a better trade-off between the avoidance of overfitting, on the one hand and sufficient depth, on the other hand. 

I shuffled and split the data into training and validation sets and observed model loss epoch by epoch for each of the sets. The fact that validation data loss was not massively higher than that on the training data, made me comfortable that overfitting was not taking place. 

For efficient use of computing power, I had initially formulated and used a generator type function for reading in data in batches. However, as I kept working on the model, Version 2 of Keras got issued, which utilized different types of parameters in its fit\_generator method. Therefore, to avoid having to re-do my code, I trained the model without generator, by reading the entire data directly in the memory. I used AWS t2.2xlarge instance, which handled the model training successfully.  
