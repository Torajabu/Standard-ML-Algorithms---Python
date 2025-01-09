# k-NN: A supervised algorithm for both classification and regression tasks.
# Note: k-NN should not be confused with k-means clustering.

# Initialization:
# - Set the value of k (number of neighbors).
# - Calculate the distance to other points in the dataset.
# - Sort points by distance to find the nearest neighbors.

# Commonly used distance metrics:
# - Euclidean distance
# - Manhattan distance

# Key characteristics of k-NN:
# - Uses majority voting for classification.
# - Optimal value of k is crucial for performance.
# - Effective for non-linear tasks.
# - Not memory efficient and slows down with an increasing number of data points.

# Techniques to address k-NN limitations:
# 1. Dimensionality Reduction: Reduces the number of features to speed up computations.
# 2. Approximate Nearest Neighbor (ANN): Trades some accuracy for faster distance computations.
# 3. Data Editing or Condensing: Reduces the dataset to critical points (e.g., points near decision boundaries)
#    to save memory and improve prediction speed.



# Step 1: Import the k-nearest neighbors algorithm from the scikit-learn package.
# Step 2: Create the feature variables and the target variables.
# Step 3: Split the data into test data and training data.
# Step 4: Generate a k-NN model using a specified number of neighbors.
# Step 5: Train the model using the training data or adjust the model based on the data.
# Step 6: Make a forecast using the trained model.


# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Loading data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)

# Train the model
knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))
