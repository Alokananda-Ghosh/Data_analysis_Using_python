#!/usr/bin/env python
# coding: utf-8

# In[146]:


from pandas import read_csv


# In[147]:


from pandas import datetime
import pandas as pd
import matplotlib.pyplot as plt
from six import StringIO
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree
import six
import sys
sys.modules['sklearn.externals.six'] = six
df = pd.read_csv("C:\\Users\\reauter\\Documents\\Data_Binning.csv")
#df.columns = ["Full_Code","Name", "Code","L.Trade", "Bid Vol","Bid","Ask","Ask Vol","Volume","High","Low","P.Close","+/- %"]
df.drop(df.columns[8:10], axis=1, inplace=True)

df


# In[148]:


from matplotlib import pyplot


# In[149]:


from pandas.plotting import autocorrelation_plot


# In[150]:


from pandas import DataFrame


# In[151]:


from statsmodels.tsa.arima_model import ARIMA


# In[152]:


df["Date"] = pd.to_datetime(df["Date"], dayfirst = True).dt.date
#Now we need to divide the Expire Date column into columns by years and months. 
#It is importanat step cause entire date is based upon year 2021
#If we divide the date into year and month we can nalyse the change in data on the basis of month
df["Year"]=pd.to_datetime(df["Date"]).dt.year
df["Month"]=pd.to_datetime(df["Date"]).dt.month


# In[ ]:





# In[ ]:





# In[153]:



import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics


# In[154]:


df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']
plt.figure(figsize=(30,100))


# In[155]:


plt.plot(df["Close"],label="Close Price History")


# In[156]:


df.drop(df.columns[0:2],df.rows[], axis=1, inplace=True)
df


# In[158]:


#Sort the dataset on date time and filter “Date” and “Close” columns:
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]


# In[160]:


df.drop(df.columns[0:2], axis=1, inplace=True)


# In[159]:


from sklearn.preprocessing import MinMaxScaler


scaler=MinMaxScaler(feature_range=(0,1))
final_dataset=new_dataset.values
train_data=final_dataset[0:100,:]
valid_data=final_dataset[100:,:]
new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)
x_train_data,y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
    
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))



# In[ ]:





# In[ ]:




