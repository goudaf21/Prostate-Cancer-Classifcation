# 297-project-5
A project using KNN's to classify prostate cancer data.

# Data Processing
We decided on a 70/30 split after testing every 5% interval between 10 and 40 for the test size and finding this optimized the accuracy. We dropped the diagnosis result to encode and append later as well as the radius. We dropped the area because we found that it had a  correlation of nearly 1 with the perimeter after running a correlation heat map of the variables. We encoded the diagnosis result so that it was readable data by the models then appended it to the data. We used a standard scalar and fit it to the training data then transformed the test and training data with the fitted standard scalar. 

# KNN
For our model of the KNN we selected a wide variety of values for n_neighbors, the weight type, algorithm, leaf size, and p then conducted a grid search to find the optimized combination of hyperparameters. We ran a confusion matrix to our classifer and found that our classifer more often classifies benign correctly than malignant which is something we could work on.