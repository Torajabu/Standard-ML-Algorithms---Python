# Stage 1: Initialise the parameters for the linear regression model.
# Stage 2: Given the value of the independent variable, predict the corresponding value of the dependent variable.
# Stage 3: Calculate the error (difference between predicted and actual value) for each data point.
# Stage 4: Perform the calculation for the partial derivatives of the linear regression (using a0 and a1).
# Stage 5: Sum up the individual costs (errors) for each of the data points.

#partial derivatives  is typically handled internally by the linear regression model. In our code, this is abstracted away and managed by the LinearRegression class during the fit() method
#our code doesn't explicitly calculate and print the error for each data point, this happens implicitly within the linear regression mode
#Similarly, the summation of errors (which leads to the cost function) is managed internally by the LinearRegression class.


# LINEAR REGRESSION

# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reading Dataset, and changing the file read location to the location of the dataset
df = pd.read_csv(r'C:\CODING\datasets\bottle.csv')
df_binary = df[['Salnty', 'T_degC']]

# Taking only the selected two attributes from the dataset
df_binary.columns = ['Sal', 'Temp']

# Renaming the columns for easier writing of the code
df_binary.head()

# Displaying only the 1st rows along with the column names
# Exploring data scatter
sns.lmplot(x='Sal', y='Temp', data=df_binary, order=2, ci=None)

# Plotting the data scatter
# Data Cleaning
# Eliminating NaN or missing input numbers
df_binary.fillna(method='ffill', inplace=True)

# Training Model
X = np.array(df_binary['Sal']).reshape(-1, 1)
y = np.array(df_binary['Temp']).reshape(-1, 1)

# Separating the data into independent and dependent variables 
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column
df_binary.dropna(inplace=True)

# Dropping any rows with NaN values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Splitting the data into training and testing data
regr = LinearRegression()
regr.fit(X_train, y_train)
print("Model score:", regr.score(X_test, y_test))

# Exploring our results
y_pred = regr.predict(X_test)

# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual values')
plt.plot(X_test, y_pred, color='black', label='Predicted values')
plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.title('Salinity vs Temperature')
plt.legend()
plt.show()

# Calculating and displaying correlation coefficient
correlation_matrix = np.corrcoef(X.T, y.T)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print("Correlation coefficient:", correlation_xy)

# Data scatter of predicted values for smaller dataset
df_binary500 = df_binary[:500]

# Selecting the 1st 500 rows of the data
df_binary500 = df_binary500.sort_values(by='Sal')
df_binary500.dropna(inplace=True)

X = np.array(df_binary500['Sal']).reshape(-1, 1)
y = np.array(df_binary500['Temp']).reshape(-1, 1)
df_binary500.fillna(method='ffill', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print("Model score for smaller dataset:", regr.score(X_test, y_test))

y_pred = regr.predict(X_test)

# Plotting the results for the smaller dataset
plt.scatter(X_test, y_test, color='green', label='Actual values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')
plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.title('Salinity vs Temperature (Smaller Dataset)')
plt.legend()
plt.show()

# Calculating and displaying correlation coefficient for the smaller dataset
correlation_matrix = np.corrcoef(X.T, y.T)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy ** 2
print("Correlation coefficient for smaller dataset:", correlation_xy)
