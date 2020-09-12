#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[5]:


cancer=load_breast_cancer()


# In[6]:


cancer.keys()


# In[9]:


print(cancer['DESCR'])


# In[10]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[11]:


df.head()


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


scale=StandardScaler()


# In[14]:


scale.fit(df)


# In[17]:


scaled_data=scale.transform(df)


# In[15]:


from sklearn.decomposition import PCA


# In[16]:


pca=PCA(n_components=2)


# In[18]:


pca.fit(scaled_data)


# In[20]:


x_pca=pca.transform(scaled_data)


# In[21]:


scaled_data.shape


# In[22]:


x_pca.shape


# In[23]:


x_pca


# In[24]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')


# In[ ]:




