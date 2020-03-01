#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd


# In[22]:


df = pd.read_csv('C:\\Users\\geetk\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\boston_house_prices.csv')


# In[15]:


df


# In[30]:


headers = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']


# In[31]:


df.columns = headers


# In[35]:


df = df[1:]


# In[36]:


df.head()


# In[37]:


#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS - proportion of non-retail business acres per town.
#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#NOX - nitric oxides concentration (parts per 10 million)
#RM - average number of rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - weighted distances to five Boston employment centres
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio by town
#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - % lower status of the population
#MEDV - Median value of owner-occupied homes in $1000's


# In[40]:


df.isnull().sum()


# In[43]:


X = df.drop("MEDV",axis=1)


# In[44]:


X.head()


# In[45]:


y = df["MEDV"]


# In[46]:


y.head()


# In[48]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# In[49]:


df_model = LinearRegression()


# In[50]:


df_model.fit(X_train,y_train)


# In[56]:


df_model.coef_


# In[57]:


predictions = df_model.predict(X_test)


# In[59]:


predictions


# In[64]:


predictions.reshape(-1, 1)


# In[65]:


y_test


# In[66]:


comparring = pd.DataFrame({
    "Actual Values":y_test,
    "predicted": predictions
})


# In[67]:


comparring


# In[82]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predictions)


# In[85]:


from sklearn.metrics import mean_squared_error
(mean_squared_error(y_test,predictions))**(1/2)


# In[ ]:




