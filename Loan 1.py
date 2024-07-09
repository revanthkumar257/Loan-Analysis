#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/DELL/Downloads/Loan Dataset/Loan Dataset/Loan 1.csv")
df


# In[26]:


df.describe(include='all')


# In[27]:


df.isnull().sum()


# In[29]:


df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df


# In[30]:


df['Income Stability'] = df['Income Stability'].fillna(df['Income Stability'].mode()[0])
df


# In[31]:


df['Type of Employment'] = df['Type of Employment'].fillna('not updated')
df.iloc[2]


# In[32]:


df['Property Location'] = df['Property Location'].fillna('not mentioned')
df.iloc[164]


# In[ ]:





# In[33]:


df['Income (USD)'] = df['Income (USD)'].fillna(df['Income (USD)'].mean())
df


# In[34]:


df['Current Loan Expenses (USD)'] = df['Current Loan Expenses (USD)'].fillna(df['Current Loan Expenses (USD)'].mean())
df


# In[35]:


df['Dependents'] = df['Dependents'].fillna(0) 
df


# In[36]:


df['Credit Score'] = df['Credit Score'].fillna(df['Credit Score'].mean())
df


# In[37]:


df['Property Age'] = df['Property Age'].fillna(df['Property Age'].mean())
df


# In[38]:


df['Has Active Credit Card'] = df['Has Active Credit Card'].fillna('blocked')
df


# In[40]:


null_counts = df.isnull().sum()
print(null_counts)


# In[22]:


summary = df.describe(include='all')
print(summary)


# In[44]:


df.hist(bins=30, figsize=(20, 15))
plt.show()


# In[43]:


plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()


# In[42]:


sns.pairplot(df)
plt.show()


# In[ ]:





# In[ ]:




