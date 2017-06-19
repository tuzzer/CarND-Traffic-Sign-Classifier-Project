**Traffic Sign Recognition** 
---


[//]: # (Image References)

[exploration]: ./write-up_images/exploration.png "exploration"
[noisy_image]: ./write-up_images/noisy_image.png "noisy_image"
[random_test_image_1]: ./write-up_images/random_test_image_1.png "random_test_image_1"
[random_test_image_2]: ./write-up_images/random_test_image_2.png "random_test_image_2"
[random_test_image_3]: ./write-up_images/random_test_image_3.png "random_test_image_3"
[new_image_1]: ./write-up_images/new_image_1.png "new_image_1"
[new_image_2]: ./write-up_images/new_image_2.png "new_image_2"
[new_image_3]: ./write-up_images/new_image_3.png "new_image_3"
[new_image_4]: ./write-up_images/new_image_4.png "new_image_4"
[new_image_5]: ./write-up_images/new_image_5.png "new_image_5"
[new_image_6]: ./write-up_images/new_image_6.png "new_image_6"
[new_image_1_train]: ./write-up_images/new_image_1_train.png "new_image_1_train"
[new_image_2_train]: ./write-up_images/new_image_2_train.png "new_image_2_train"
[new_image_3_train]: ./write-up_images/new_image_3_train.png "new_image_3_train"
[new_image_4_train]: ./write-up_images/new_image_4_train.png "new_image_4_train"
[new_image_5_train]: ./write-up_images/new_image_5_train.png "new_image_5_train"
[new_image_6_train]: ./write-up_images/new_image_6_train.png "new_image_6_train"
[bicycle_train]: ./write-up_images/bicycle_train.png "bicycle_train"


Here is a link to my [project code](https://github.com/tuzzer/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).


###Data Set Summary & Exploration

![alt text][exploration]

I plotted the number of samples in each class in the training set in descending order.

The number of samples among the different classes is very unbalanced.
This can cause overfitting with the over-represented class and it might not be able to predict under-represented classes well. 

To alleviate this problem, I oversampled the under-represented classes with added noise.

###Design and Test a Model Architecture

####1. Pre-process the Data Set

Here I oversampled the training samples to make all classes so that each class has the sample number of samples. 

I added "salt and pepper" noise to the additional samples to reduce over-fitting. Note that noisy images are added to the classes with most samples as well.
![alt text][noisy_image]

In addition, I greyscaled the images to reduce the number of features. This allows the use of neural network with fewer parameters and requires fewer training samples

After pre-processing, the number of training samples became 103716, 
and the number of color channel became 1.


####2. Model Architecture

I modified the LeNet used in the MNIST example. 
The main change I made was to increase the depth of the first two convolutional layers to 32 and 64
respectively to account for the added complexity of the data. 
I also increased the number of epochs to 150 since there are more training samples and neurons (when compared to MNIST).
Moreover, I added a dropout layer to reduce over-fitting.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					|
| Convolution 3x3     	| 1x1 stride, valid padding, depth 32 	        |
| RELU					|												|
| Max pooling	      	| 2x2 stride				                    |
| Convolution 5x5	    | 1x1 stride, valid padding, depth 64     		|
| RELU					|												|
| Max pooling	      	| 2x2 stride 				                    |
| Fully connected		| output 120        						    |	
| RELU					|												|
| Fully connected		| output 84        						        |	
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| output 43        						        |	


####3. Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. 
A low accuracy on the training and validation sets imply underfitting. 
A high accuracy on the training set but low accuracy on the validation set implies overfitting.


After training for 150 epochs, the accuracy on the validation set was approximately 0.962 and the accuracy on the test set was approximately 0.943. 
The model slightly over-fitted but the difference is relatively small.

#####Prediction on randomly selected test images

######Image # 0

![alt text][random_test_image_1]

Prediction = Speed limit (80km/h)


######Image # 1

![alt text][random_test_image_2]

Prediction = General caution

######Image # 2

![alt text][random_test_image_3]

Prediction = Speed limit (50km/h)

#### 4. Discussions
##### What was the first architecture that was tried and why was it chosen?
LeNet used in MNIST example was used. It was chosen because it was already implemented and the two problems are quite similar.

##### What were some problems with the initial architecture?
It didn't have a dropout layer which may lead to overfitting. 

##### How was the architecture adjusted and why was it adjusted?
A dropout layer was added for the reason mentioned above. 

#### Which parameters were tuned? How were they adjusted and why?
The initial architecture has too few neurons and was unable to capture the complexity of the data set effectively. 
Additional neurons were added to the convolutional layer.
Moreover, a larger number of epochs was used to train the network since there are more training data and the data is
inherently more complex. 

#### What are some of the important design choices and why were they chosen?
Under-represented data was resampled to balance out the data set. 
Additional noise was added to the images to reduce overfitting.


###Test a Model on New Images

Here are six German traffic signs that I found on the web:

![alt text][new_image_1] ![alt text][new_image_2] 
![alt text][new_image_3] ![alt text][new_image_4] 
![alt text][new_image_5] ![alt text][new_image_6]


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycle Crossing		| Roadwork   					    			| 
| Wild animals crossing | Wild animals crossing 						|
| Go straight or right 	| Go straight or right          				|
| Pedestrians      		| Right-of-way at the next intersection			|
| Road work 			| Road work           							|
| Stop         			| Stop                 							|


Out of the 6 new images that I found on the web, I got 4 of them right. 

This gives the model an accuracy of 0.66 for new images. This is low compared to the accuracy on the test set. 


####3. Prediction Confidence

The ones that my model got wrong are "Pedestrian", and "Bicycle Crossing". The confidence on the incorrectly classified images are relatively low compared to the ones that were correct. This might be because the images in the test set are more similar to the training set. 

For "Pedestrians", the second most probable prediction was correct. This might be attribute to the similarity between the "Right-of-way" and "Pedestrians" signs.

The images below showed an example of the "Bicycle Crossing" class in our training set. Due to small size and fine details in the images, a 32x32 sample is very pixelated and it is even difficult for human being to tell what the signs are. This might have contributed to the incorrect predictions. 

######Bicycle Crossing in training sample

The images below showed an example of the this class in our training set.

![alt text][bicycle_train]

Due to small size and fine details in the images, a 32x32 sample is very pixelated and it is even difficult for human being to tell what the signs are.
This might have contributed to the incorrect predictions. 


Below are the top five predictions for each of the images and their associated softmax probabilities.

#####Bicycle Crossing		
        			
    Top 5 Predictions
        Road work -- 1.92624e-08
        Slippery road -- 1.59267e-08
        Bicycles crossing -- 7.75884e-09
        Ahead only -- 4.04294e-13
        Dangerous curve to the right -- 2.06181e-13

#####Wild animals crossing 

    Top 5 Predictions
        Wild animals crossing -- 0.095946
        Slippery road -- 1.35823e-09
        Dangerous curve to the left -- 8.174e-10
        Bicycles crossing -- 4.1841e-12
        General caution -- 7.35425e-15
     

#####Go straight or right 	

    Top 5 Predictions
        Go straight or right -- 0.900367
        Speed limit (60km/h) -- 1.18693e-15
        Ahead only -- 6.63823e-17
        Keep right -- 3.24098e-18
        Yield -- 6.54349e-19

 
#####Pedestrians      		

    Top 5 Predictions
	    Right-of-way at the next intersection -- 0.00333398
        Pedestrians -- 0.000179032
        General caution -- 0.000172521
        Double curve -- 2.65362e-16
        Speed limit (100km/h) -- 7.09493e-19
	 
#####Road work

    Top 5 Predictions 
        Road work -- 9.3428e-07
        Speed limit (60km/h) -- 2.36687e-11
        Slippery road -- 2.00948e-11
        General caution -- 1.28413e-11
        Keep right -- 9.78972e-12

#####Stop

    Top 5 Predictions
        Stop -- 1.76267e-08
        Turn right ahead -- 9.30297e-10
        Keep right -- 7.33288e-10
        Speed limit (50km/h) -- 6.77346e-10
        No entry -- 3.93038e-10
	 
