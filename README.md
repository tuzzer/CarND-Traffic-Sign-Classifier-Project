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
[new_image_5]: ./write-up_images/write-up_images/new_image_5.png "new_image_5"
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


###Test a Model on New Images

Here are six German traffic signs that I found on the web:

![alt text][new_image_1] ![alt text][new_image_2] 
![alt text][new_image_3] ![alt text][new_image_4] 
![alt text][new_image_5] ![alt text][new_image_6]


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycle Crossing		| Ahead only    								| 
| Wild animals crossing | Wild animals crossing 						|
| Go straight or right 	| Go straight or right          				|
| Pedestrians      		| Pedestrians					 				|
| Road work 			| Road work           							|
| Stop         			| Stop                 							|


Out of the 6 new images that I found on the web, I got 5 of them right. 

This gives the model an accuracy of 0.83 for new images. This is low compared to the accuracy on the test set. 


####3. Prediction Confidence

The one that my model got wrong is "Bicycle Crossing". The confidence on the incorrectly classified images are relatively low. 
Samples in the test set might be relative similar which caused the overfitting.

######Bicycle Crossing in training sample

The images below showed an example of the this class in our training set.

![alt text][bicycle_train]

Due to small size and fine details in the images, a 32x32 sample is very pixelated and it is even difficult for human being to tell what the signs are.
This might have contributed to the incorrect predictions. 


#####Bicycle Crossing		
        			
Top 5 Predictions
	 Ahead only -- 1.14582
	 Bicycles crossing -- -0.354429
	 Children crossing -- -4.57259
	 Slippery road -- -6.26757
	 Speed limit (60km/h) -- -6.53344

#####Wild animals crossing 

Top 5 Predictions
	 Wild animals crossing -- 5.51297
	 Slippery road -- -0.466942
	 No passing for vehicles over 3.5 metric tons -- -9.11336
	 Dangerous curve to the left -- -12.5807
	 Road work -- -15.5678
 

#####Go straight or right 	

Top 5 Predictions
	 Go straight or right -- 17.6862
	 Dangerous curve to the left -- -18.5981
	 Road work -- -18.6661
	 General caution -- -22.9936
	 Bicycles crossing -- -23.2801

 
#####Pedestrians      		

Top 5 Predictions
	 Pedestrians -- 26.7529
	 Right-of-way at the next intersection -- 15.7218
	 General caution -- -8.36611
	 Roundabout mandatory -- -21.1614
	 Vehicles over 3.5 metric tons prohibited -- -25.2543
	 
#####Road work

Top 5 Predictions 
	 Road work -- 15.2368
	 Dangerous curve to the right -- -8.52853
	 Beware of ice/snow -- -19.1516
	 No passing for vehicles over 3.5 metric tons -- -28.595
	 Wild animals crossing -- -30.0119

#####Stop

Top 5 Predictions
	 Stop -- 2.01156
	 Speed limit (30km/h) -- -4.08149
	 Speed limit (70km/h) -- -4.47525
	 Turn right ahead -- -4.87525
	 Roundabout mandatory -- -6.19679

