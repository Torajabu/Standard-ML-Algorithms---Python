# Standard ML Algorithms - Python

This repository contains my local Python implementations of various standard machine learning algorithms. These implementations utilize libraries such as Scikit-learn, Matplotlib, and NumPy.

## About
This repository provides a collection of Python scripts that demonstrate the implementation of machine learning algorithms. The algorithms are implemented using various Python libraries.

**Note:** I have also attached the results of running most of these algorithms locally

## Algorithms Included

### Classification Algorithms
- **Naive Bayes:** A classification method based on Bayes' theorem, assuming predictors are independent. The implementation involves importing a dataset, calculating prior probabilities, determining likelihoods, and calculating posterior probabilities.
  
- **K-Nearest Neighbors (KNN):** A classification algorithm that classifies data points based on the majority class of their k-nearest neighbors. The implementation includes importing the algorithm, creating feature and target variables, splitting data, creating a KNN model, training the model, and making predictions.
  
- **Decision Trees:** A supervised learning algorithm used for both classification and regression. The implementation involves creating a node for each attribute, with the most significant at the top, and recursively distributing records.
  
- **Logistic Regression (LR):** A classification algorithm used to predict the probability of a binary outcome. It is a discriminative model, unlike Naive Bayes, and performs well even with collinearity.
  
- **Support Vector Machines (SVM):** A supervised learning technique for classification. The implementation steps include importing libraries, loading the dataset, dividing the dataset into X and Y, creating training and test sets, scaling features, adjusting the SVM to the training set, making predictions, and creating a confusion matrix.

### Regression Algorithms
- **Linear Regression:** A model used to determine the relationship between independent and continuous dependent variables. The implementation involves initializing parameters, predicting dependent variable values, determining errors, and calculating partial derivatives.
  
- **Polynomial Regression:** A type of linear regression where the relationship between variables is modeled as an nth degree polynomial. Polynomial terms are added to transform linear regression into polynomial regression.

### Feature Selection and Extraction
- **Principal Component Analysis (PCA):** A technique used for dimensionality reduction by generating a set of primary variables.

### Association Rule Learning
- **Apriori Algorithm:** A data mining technique for finding frequent item sets and association rules. The implementation involves generating candidate sets, testing subsets, and calculating the final frequent itemset.

### Clustering Algorithms
- **K-Means Clustering:** An algorithm used for partitioning data into k clusters.

## Repository Structure
The repository includes the following files:
- apriori algorithm1st
- k means clustering results
- linear regression dataset and output
- pca output
- polynomial regression output
- README.md
- SUPPORT VECTOR MACHINE
- decision trees.py
- k-means clustering.py
- knn.py
- linear regression.py
- logistic regression
- naive bayes.py
- polynomial regression.py
- principal component analysis.py

## Usage
To use the code, you will need Python along with libraries such as Scikit-learn, Matplotlib, and NumPy installed. Each script implements a specific machine learning algorithm. The implementation of each algorithm is discussed in the sources.

## Further Reading
- [Kaggle](https://www.kaggle.com)
- [Github](https://github.com)
- [Towards Data Science](https://towardsdatascience.com)
- [Machine Learning Mastery](https://machinelearningmastery.com)
