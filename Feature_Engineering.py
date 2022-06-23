#!/usr/bin/env python
# coding: utf-8

# In[92]:


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
df = pd.read_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\NFBNIFC12.csv")
#df.columns = ["Full_Code","Name", "Code","L.Trade", "Bid Vol","Bid","Ask","Ask Vol","Volume","High","Low","P.Close","+/- %"]
#df.drop(df.columns[7:9], axis=1, inplace=True)
df


# In[25]:


df.set_index('Date')


# In[26]:


df.loc[2]


# In[16]:


#df['Time'] = np.where(df['Time']=='15:30:00',df['Close'], df['Close-Open'])


# In[17]:


#df['Time']


# In[27]:


df['Date']


# In[46]:


df['Date'] = pd.to_datetime(df['Date'])


# In[47]:


df


# In[48]:


#To paste this dataframe to csv file
df.to_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\NFBNIFC12.csv", index=False)


# In[51]:


#Extracting Monthly data from date format
df['Month'] = pd.to_datetime(df.Date, format='%d/%m/%Y').dt.month_name()


# In[53]:


df['Month']


# In[57]:


#Determining 7 times of a day
df['times_of_day'] = pd.to_datetime(df.Time, format='%H:%M:%S')
a = df.assign(dept_session=pd.cut(df.times_of_day.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
df['times_of_day'] = a['dept_session']


# In[58]:


df['times_of_day']


# In[60]:


#12 Dimensionality Reduction Techniques (with Python codes)
#Low Variance Filter
df.var()


# In[ ]:


#he variance of <=100 column is very low.Sp we can safely drop this column.


# In[61]:


#High Correlation filter
df.corr()


# In[ ]:


#There is high correlation among HIGH,LOW,CLOSE,OPEN column.Dropping these columns would be prudent.
#But this domains are important for trading as well.


# In[63]:


df


# In[67]:


train=df.drop(df.columns[0:1], axis=1, inplace=True)
df
train=df.drop(df.columns[5:6], axis=1, inplace=True)
df


# In[72]:





# In[70]:


df2=df.dropna().reset_index(drop=True)


# In[71]:


df2


# In[80]:


#RandomForest
from sklearn.ensemble import RandomForestRegressor
#df=df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
model = RandomForestRegressor(random_state=1, max_depth=10)
df=pd.get_dummies(df)
model.fit(df2,df2.Close)


# In[81]:


#For RandomForest there should be only quantitative data and there couldn't be any NaN in the data.
#After fitting the model, plot the feature importance graph:


# In[82]:


features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[84]:


#, we can use the SelectFromModel of sklearn to do so. 
from sklearn.feature_selection import SelectFromModel
feature = SelectFromModel(model)
Fit = feature.fit_transform(df2,df2.Close)


# In[85]:


#Forward Feature Selection
from sklearn.feature_selection import f_regression
ffs = f_regression(df2,df2.Close )


# In[87]:


variable = [ ]
for i in range(0,len(df.columns)-1):
    if ffs[0][i] >=10:
       variable.append(df.columns[i])


# In[88]:


variable


# In[93]:


#Factor Analysis
df


# In[95]:


df.drop(df.columns[0:1], axis=1, inplace=True)
df


# In[98]:


df.drop(df.columns[6:7], axis=1, inplace=True)
df


# In[101]:


'''
Adequacy Test
Before you perform factor analysis, you need to evaluate the “factorability” of our dataset.
Factorability means "can we found the factors in the dataset?".
There are two methods to check the factorability or sampling adequacy:
'''


# In[108]:


df2
df2.


# In[119]:


df2.drop(df2.columns[2:3], axis=1, inplace=True)
df2


# In[ ]:





# In[ ]:





# In[ ]:





# In[134]:


df


# In[124]:


df.drop(df.columns[5:7], axis=1, inplace=True)
df


# In[127]:


df=df.dropna().reset_index(drop=True)


# In[129]:


#Bartlett’s test 
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value


# In[132]:


#KMO TEST
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)


# In[133]:


kmo_model


# In[135]:


'''
Since the value of the kmo model is more than 0.6 so 
we can say the model is adequate for factor analysis
'''


# In[ ]:





# In[165]:


fa = FactorAnalyzer(2, rotation="varimax", method='minres', use_smc=True)
fa.fit(df)


# In[ ]:





# In[166]:


fa.loadings_


# In[167]:


fa.get_communalities()


# In[168]:


loadings = pd.DataFrame(fa.loadings_, columns=['Factor 1', 'Factor 2'], index=df.columns)
print('Factor Loadings \n%s' %loadings)


# In[169]:


'''
Start with the Varimax rotation

The method can be set as minres, ml or principal.We can start to minres, while performing Varimax rotation.

Change the method to maximum likelihood but still use Varimax rotation. 

Two logical choices are available for whether to use squared multiple correlation as starting guesses for factor analysis. Always start with smc (e.g. squared multiple correlation) and try maximum absolute correlation as second.  We can specify this by setting use_smc=True. 

Compare the solutions and keep the one that works the best.

Evaluate factor loadings and consider a different factor solution: one higher and one lower than the chosen k (in our current case four).  

If we partition the data, we can now try the solution on test data. 
'''


# In[170]:


'''
Loadings close to -1 or 1 indicate that the factor strongly influences the variable.
Loadings close to 0 indicate that the factor has a weak influence on the variable.
Some variables may have high loadings on multiple factors.
Unrotated factor loadings are often difficult to interpret.
'''


# In[ ]:


'''
Open,High,Low,Close are creating Factor 1 and Volume is creating factor 2.
'''


# In[ ]:




