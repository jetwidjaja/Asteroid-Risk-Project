# Asteroid-Risk-Project

This project is based on https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification.

In this project,  our goal is to use Machine Learning binary classification to categorize whether an asteroid is hazardous or not. The data will be taken from NASA's Small-Body databse which contains data about approximately 32000 celestial objects close to Earth. I have chosen 42 of the variables to investigate, excluding all the irrelevant ones to the Machine Learning model I will be implementing such as asteroid ID. I uploaded the queried dataset with relevant variables onto https://git.ucsc.edu/jwidjaj1/phys-152/-/raw/master/HW3/asteroid_data.csv.

The data is extremely biased, with approximately 90% of the asteroids classified as non-hazardous. We therefore implemented undersampling, shrinking our data to only 4488 data points with an equal number of hazardous and non-hazardous asteroids. We reassigned the Yes and No values under the hazardous variable to 0 and 1 respectively.

The data contains missing values, with some variables not having a value in approximately 90% of the data points. We will eliminate these variables since due to the lack of data. We will then impute the remaining missing data for variables with fewer missing points by assigning them with the median values of their respective variable. Finally, we will do a 80/20 split into training and testing data sets.

We will be using the aritifical neural network(ANN) machine learning model with 3 hidden layers 64, 128 and 32 nodes in that order, all with the ReLU activation function. We will be using the Sigmoid activation function for the output layer to obtain an output from 0 to 1 to represent the probability of the asteroid being hazardous. We have chosen the nn.BCEWithLogitsLoss() loss function which is the most favorable for a binary classification problem and the Adam optimizer with a learning rate of 0.0001.

We used minibatch training with a batch size of 32 and ran it through 500 epochs. After checking it against the test dataset, we obtained an accuracy of 95.43% and the ROC score is 0.9546.
