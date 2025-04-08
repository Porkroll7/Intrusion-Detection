#The Scikit-Learn API is designed with the following guiding principles in mind, as outlined in the Scikit-Learn API paper:
#Consistency, Inspection, Limited object hierarchy, Composition, Sensible defaults:

#Most commonly, the steps in using the Scikit-Learn Estimator API are as follows:
#Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
#Choose model hyperparameters by instantiating this class with desired values.
#Arrange data into a features matrix and target vector, as outlined earlier in this chapter.
#Fit the model to your data by calling the fit method of the model instance.
#Apply the model to new data:
#For supervised learning, often we predict labels for unknown data using the predict method.
#For unsupervised learning, we often transform or infer properties of the data using the transform or predict method.
#We will now step through several simple examples of applying supervised and unsupervised learning methods.




# Code Created by:  Kyle Ketterer
# Date: 04/07/2025

import numpy as np
from sklearn.linear_model import LinearRegression


#logistic regression
model = LinearRegression()
