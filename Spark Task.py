#!/usr/bin/env python
# coding: utf-8

# THE SPARKS FOUNDATION
# 
# 

# DATA SCIENCE AND BUSINESS ANALYTICS

# INTERN NAME - POOJA RAMESH BARHATE

# TASK 01 - Prediction using supervised ML
# 
# 

# Statement :  To predict the percentage of an student based on the no. of study hours.What will be predicted score if a student                studies for 9.25 hrs/ day?

# In[2]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#importing the given data
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")


# In[4]:


s_data.head(10)


# In[5]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[6]:



#Preparing the data and dividing it into attributes and labels.
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[7]:


print("Training complete.")
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_


# In[8]:


# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()
print(X_test) 


# In[9]:


# Testing data - In Hours
y_pred = regressor.predict(X_test) 


# In[10]:


# Predicting the scores
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[11]:


#Evaluating the model to find the mean error.
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:





# In[ ]:




