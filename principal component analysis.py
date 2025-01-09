# Feature selection and extraction are one of the most important steps that must be performed in order for machine learning to be successful. let us start the implementation part of the mechanisms that we have learned for 
# Feature selection and extraction. Let's begin with dimensionality reduction, the process of lowering the number of random variables that are being considered 
# by generating a set of primary variables. Dimensionality reduction may be seen as a way to streamline the analysis process.

# Among the various techniques, Principal Component Analysis (PCA) is the most 
# frequently used, and the implementation of PCA is given below:

#for the dataset used , refer to https://www.kaggle.com/datasets/uciml/mushroom-classification
#for the outputs of this algorithm, see https://github.com/Torajabu/Standard-ML-Algorithms---Python/tree/main/pca%20output

# PRINCIPAL COMPONENT ANALYSIS (PCA)
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
m_data = pd.read_csv(r'path to mushrooms.csv')

# Machine learning systems work with integers, we need to encode these
# string characters into ints
encoder = LabelEncoder()

# Now apply the transformation to all the columns:
for col in m_data.columns:
    m_data[col] = encoder.fit_transform(m_data[col])

# Feature selection
X_features = m_data.iloc[:, 1:23]
y_label = m_data.iloc[:, 0]

# Scale the features
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)

# Visualize
pca = PCA()
pca.fit_transform(X_features)
pca_variance = pca.explained_variance_
plt.figure(figsize=(8, 6))
plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

pca2 = PCA(n_components=17)
pca2.fit(X_features)
X_3d = pca2.transform(X_features)
plt.figure(figsize=(8, 6))
plt.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=m_data['class'])
plt.show()
