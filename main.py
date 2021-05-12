# Fady Gouda and Griffin Noe
# CSCI 297a
# Project 5
# 10/11/20

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Import data from csv
data=pd.read_csv('prostate-cancer-prediction.csv')

# Split the data set into a 70/30 train/test and drop the diagnosis result
dropped = ['diagnosis_result', 'area']
X_train, X_test, y_train, y_test= train_test_split(data.drop(dropped,axis=1),data['diagnosis_result'],test_size=0.3,random_state=1)

# Instantiate the label encoder and encode the diagnosis result then append to the data
le = preprocessing.LabelEncoder()
le.fit(data['diagnosis_result'])
data['diagnosis_result']=le.transform(data['diagnosis_result'])

# Instantiate the standard scalar and fit it to the training data
sc=StandardScaler()
sc.fit(X_train)

# Transform the train and test data using the fitted standard scalar
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

# Instantiate the K-Nearest Neighbors Classifier with the selected hyperparameters 
KNN=KNeighborsClassifier(n_neighbors=11,weights='distance',algorithm='auto',leaf_size=10,p=1,metric="minkowski",metric_params=None,n_jobs=None)

# Fit the KNN to the training data and labels
KNN.fit(X_train_std,y_train)

# Feed the scaled test data into the KNN for prediction 
prediction = KNN.predict(X_test_std)

# Function to return the accuracy between predicted values and actual values
def accuracy(y_values,prediction):
    comparison = tf.equal(prediction, y_values)
    acc = tf.reduce_mean(tf.cast(comparison, tf.float32))
    return acc.numpy()

# Print the accuracy for the KNN model on the test data and the prediction

# Create the parameter grid with the selected permutations of hyperparameters
param_grid = [
  {'n_neighbors': [1,3,5,7,9,11], 'weights': ['distance', 'uniform'],'algorithm':['auto','ball_tree','kd_tree','brute'],'leaf_size':[10,20,30],'p':[1,2]}
 ]

# Instantiate the grid search with the parameter grid
clf=GridSearchCV(KNeighborsClassifier(),param_grid,verbose=0)

# Get the KNN with the grid search optimized hyperparameters 
clf_results=clf.fit(X_train_std,y_train)

# Feed the scaled test data into the grid search optimized KNN 
pred=clf.predict(X_test_std)


# Print the accuracies for both of the KNNs
print("KNN Accuracy: %.3f" % accuracy(y_test,pred))
print("Grid Search Parameters: ", clf_results.best_params_)

# Instantiate and show the confusion matrix of the KNN with the scaled test data and display it
plot_confusion_matrix(clf,X_test_std,y_test)
plt.show()