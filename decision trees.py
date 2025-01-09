#root node>sub node>decision node>leaf node, prone to overfitting,    gini impurity#
#Ensemble methods leverage the wisdom of crowds by combining multiple models, which generally leads to better prediction accuracy and reduced overfitting compared to a single decision tree.
#Hyperparameter tuning is crucial for both individual trees and ensembles to balance between underfitting and overfitting.
#Cross-validation should be used to ensure the robustness of the chosen parameters across different data samples
#used for both regression and classification tasks##


# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load the Iris dataset directly from sklearn
iris = load_iris()
X, y = iris.data, iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y

# Display first few rows of the dataset
print(df.head())

# Check dataset shape
print("Shape of dataset:", df.shape)

# Information about the dataset
print(df.info())

# Check for null values
print(df.isnull().sum())

# Display summary statistics
print(df.describe().T)

# Check for outliers using box plots
for col in iris.feature_names:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot for {col}')
    plt.show()

# Note: Here we would typically remove outliers, but for simplicity, we'll proceed without this step.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize and train the Decision Tree classifier with some default parameters
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=1)
dt.fit(X_train, y_train)

# Visualize the decision tree (this requires graphviz to be installed)
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()

# Evaluate the model on training and test data
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

print('Accuracy of Decision Tree on Training Data:', accuracy_score(y_train, y_pred_train))
print('Accuracy of Decision Tree on Test Data:', accuracy_score(y_test, y_pred_test))

# Print classification report for the test data
print('Classification Report for Test Data:\n', classification_report(y_test, y_pred_test))

# Hyperparameter Tuning using GridSearchCV
dt = DecisionTreeClassifier(random_state=1)
param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

grid_search = GridSearchCV(dt, param_grid, cv=3)
grid_search.fit(X, y)

# Best parameters from GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Train with the best parameters
best_dt = DecisionTreeClassifier(**grid_search.best_params_, random_state=1)
best_dt.fit(X_train, y_train)

# Predict with the optimized model
y_pred_best_train = best_dt.predict(X_train)
y_pred_best_test = best_dt.predict(X_test)

# Evaluate the optimized model
print('Accuracy of Optimized Decision Tree on Training Data:', accuracy_score(y_train, y_pred_best_train))
print('Accuracy of Optimized Decision Tree on Test Data:', accuracy_score(y_test, y_pred_best_test))

# Confusion Matrix for the optimized model
print('Confusion Matrix - Train:\n', confusion_matrix(y_train, y_pred_best_train))
print('\nConfusion Matrix - Test:\n', confusion_matrix(y_test, y_pred_best_test))



# Print classification report for the optimized model on test data
print('Classification Report for Optimized Model on Test Data:\n', classification_report(y_test, y_pred_best_test))
#Uses statistical methods for feature selection implicitly within the DecisionTreeClassifier for constructing the tree, but this process isn't directly shown or manipulated in the code.
#Performs hyperparameter tuning which indirectly affects which features might be selected for splits at different level#
