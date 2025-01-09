#In the first step, we will begin by importing the dataset as well as any necessary dependencies...
#The second step is to get the prior probability of each class using the formula P(y).
# The Third Step is to Determine the likelihood of each characteristic using the table you just created...
#Final and the Fourth Step is to Calculate the Posterior Probability for each class by applying the Naive Bayesian equation.#

# Load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# Splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Training the model on the training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = gnb.predict(X_test)

# Comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy (in %):", metrics.accuracy_score(y_test, y_pred) * 100)

#One benefit is the NB models have low variance. Hence avoid overfitting.
#since it  assumes all attributes as independent, it gives all attributes equally(equal weight)
#in nlp, we use naive bayes
#in linear regression, one attribute might be more powerful, i.e output might change depending on one single attribute
#Naive Bayes classifiers are particularly useful when you don't have a lot of data#
