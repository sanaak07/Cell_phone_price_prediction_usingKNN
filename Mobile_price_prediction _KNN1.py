#!/usr/bin/env python
# coding: utf-8

# # MOBILE PRICE CLASSIFICATION USING KNN

# In[1]:


import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import required data file
data = pd.read_csv('Cellphone.csv')


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


print(data.Price.value_counts())


# In[8]:


sns.countplot(x='Price',data = data, palette='RdBu_r')


# In[9]:


sns.scatterplot(y='Price',x='ram', data = data)


# In[10]:


sns.boxplot(x='Price', y='battery', data = data)


# In[11]:


sns.countplot(x='Price', hue='resoloution', data=data, palette='RdBu_r')


# In[12]:


sns.distplot(data['internal mem'], kde=False, color='darkred', bins=55)


# In[13]:


data.head()


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


#split data into test and train data
X_train, X_test, y_train, y_test = train_test_split (data.drop('Price', axis = 1),
                                                    data['Price'], test_size=0.33,
                                                    random_state=101)


# In[16]:


#Create KNN model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)


# In[17]:


#Finding suitable n_neighbors value using elbow method
knn.score(X_test, y_test)


# In[18]:


preds = knn.predict(X_test)


# In[19]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[20]:


print(classification_report(preds, y_test))


# In[21]:


print(accuracy_score(preds, y_test))


# In[22]:


matrix = confusion_matrix (y_test, preds)
print (matrix)


# # CONCLUSION

# The accuracy we got is around 0.018 using the KNN algorithm
