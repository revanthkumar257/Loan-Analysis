#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

# Load the dataset
dataa = pd.read_csv("C:/Users/DELL/OneDrive/Desktop/Loan Analysis/Loan 2.csv")

# Handle missing values
categorical_columns = ['Type of Employment', 'Income Stability', 'Has Active Credit Card', 'Property Location', 'Gender']
for column in categorical_columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

numerical_columns = ['Property Age', 'Income (USD)', 'Dependents', 'Credit Score', 'Loan Sanction Amount (USD)', 'Current Loan Expenses (USD)']
for column in numerical_columns:
    data[column].fillna(data[column].median(), inplace=True)

# Function to remove outliers using IQR method
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Numerical columns to check for outliers
numerical_columns = ['Age', 'Income (USD)', 'Loan Amount Request (USD)', 'Credit Score', 'Property Age', 'Property Price', 'Loan Sanction Amount (USD)', 'Current Loan Expenses (USD)']

# Remove outliers
data_cleaned = remove_outliers(data, numerical_columns)

# Save the cleaned data
data_cleaned.to_csv('cleaned and updated loandataset.csv', index=False)

# Check the shape of the dataset before and after removing outliers
print("Shape before removing outliers:", dataa.shape)
print("Shape after removing outliers:", data_cleaned.shape)


# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.cluster import KMeans


data = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Loan Analysis\cleaned and updated loandataset.csv")

data


# In[23]:


# Feature Engineering (if any additional features needed)
# Example: Creating a new feature 'Loan to Income Ratio'
data_cleaned['Loan to Income Ratio'] = data_cleaned['Loan Amount Request (USD)'] / data_cleaned['Income (USD)']


# In[18]:


# EDA
# Univariate Analysis
plt.figure(figsize=(10, 6))
data_cleaned['Age'].hist(bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[19]:


# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Income (USD)',data=data)
plt.title('Income Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Income (USD)')
plt.show()


# In[20]:


# Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(data_cleaned.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[21]:


# Machine Learning Tasks
# 1. Fraudulent Loan Application Detection (Classification)
# For the sake of this example, let's assume we have a target column 'Fraud' indicating fraudulent applications
# Split the data
X = data_cleaned.drop(columns=['Fraud'])
y = data_cleaned['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[29]:


pip install numpy pandas scikit-learn matplotlib seaborn


# In[31]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assuming 'data' is your DataFrame
# Print the column names to check if 'Loan Sanction Amount (USD)' is present
print(data.columns)

# Verify the DataFrame structure
print(data.head())

# Ensure 'Loan Sanction Amount (USD)' column exists before splitting the data
if 'Loan Sanction Amount (USD)' in data.columns:
    # Convert categorical columns to numerical values using one-hot encoding
    data_encoded = pd.get_dummies(data.drop(columns=['Customer ID', 'Name']))
    
    # Split the data
    X = data_encoded.drop(columns=['Loan Sanction Amount (USD)'])
    y = data_encoded['Loan Sanction Amount (USD)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest Regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Evaluate the model
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
else:
    print("Column 'Loan Sanction Amount (USD)' not found in DataFrame")


# In[30]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data_cleaned' is your DataFrame
# Convert categorical columns to numerical values using one-hot encoding
data_cleaned_encoded = pd.get_dummies(data_cleaned.drop(columns=['Customer ID', 'Name']))

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data_cleaned['Cluster'] = kmeans.fit_predict(data_cleaned_encoded)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income (USD)', y='Loan Amount Request (USD)', hue='Cluster', data=data_cleaned, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Income (USD)')
plt.ylabel('Loan Amount Request (USD)')
plt.show()


# In[33]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming 'data_cleaned' is your DataFrame
# Print the column names to check if 'Repaid' is present
print(data_cleaned.columns)

# Verify the DataFrame structure
print(data_cleaned.head())

# Ensure 'Repaid' column exists before splitting the data
if 'Repaid' in data_cleaned.columns:
    # Convert categorical columns to numerical values using one-hot encoding
    data_encoded = pd.get_dummies(data_cleaned.drop(columns=['Customer ID', 'Name']))
    
    # Split the data
    X = data_encoded.drop(columns=['Repaid'])
    y = data_encoded['Repaid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
else:
    print("Column 'Repaid' not found in DataFrame")


# In[35]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming 'data_cleaned' is your DataFrame
# Print the column names to check if 'Approved' is present
print(data_cleaned.columns)

# Verify the DataFrame structure
print(data_cleaned.head())

# Ensure 'Approved' column exists before splitting the data
if 'Approved' in data_cleaned.columns:
    # Convert categorical columns to numerical values using one-hot encoding
    data_encoded = pd.get_dummies(data_cleaned.drop(columns=['Customer ID', 'Name']))
    
    # Split the data
    X = data_encoded.drop(columns=['Approved'])
    y = data_encoded['Approved']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
else:
    print("Column 'Approved' not found in DataFrame")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




