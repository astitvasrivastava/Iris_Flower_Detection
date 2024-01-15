#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


df=pd.read_csv('C:/Users/astit/Downloads/Iris (2).csv')


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.info


# In[10]:


df.isnull().sum()


# In[11]:


df.Species.value_counts


# In[12]:


X= df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= df['Species']


# In[14]:


X


# In[15]:


y


# In[16]:


# Do the train/test split


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# In[18]:


# Training the Linear Regression Model


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


# Let's create an instance for the LogisticRegression model


# In[22]:


lr = LogisticRegression()


# In[23]:


# Train the model on our train dataset


# In[24]:


lr.fit(X,y)


# In[25]:


# Train the model with the training set


# In[26]:


lr.fit(X_train,y_train)


# In[27]:


# Getting predictions from the model for the given examples.


# In[28]:


predictions = lr.predict(X)


# In[29]:


# Compare with the actual charges


# In[30]:


Scores = pd.DataFrame({'Actual':y,'Predictions':predictions})
Scores.head()


# In[31]:


y_test_hat=lr.predict(X_test)


# In[32]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat)*100,'%')


# In[ ]:




