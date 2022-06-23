#!/usr/bin/env python
# coding: utf-8

# In[70]:


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
df.drop(df.columns[0:2], axis=1, inplace=True)
df.drop(df.columns[6:212], axis=1, inplace=True)
df


# In[2]:


#Regression – Estimating the relationships between variables by optimizing the reduction of error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# In[18]:


df.describe()


# In[30]:


'''
Quick takeaways: We are working with a data set that contains 277 observations,
mean Opening price is approximately Rs.34438, median price is approximately Rs.34822
'''
'''
fig.add_subplot(221)   #top left
fig.add_subplot(222)   #top right
fig.add_subplot(223)   #bottom left
fig.add_subplot(224)   #bottom right
'''

fig = plt.figure(figsize=(16, 6))
open = fig.add_subplot(221)
close = fig.add_subplot(224)

open.hist(df.Open, bins=80)
open.set_xlabel('Open')
open.set_title("Histogram of Openning price")

close.hist(df.Close, bins=80)
close.set_xlabel('Close')
close.set_title("Histogram of Closing Prices")

plt.show()


# In[33]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[36]:


'''
When you code to produce a linear regression summary with OLS with only two variables this will be the formula that you use:
The “Ordinary Least Squares” module will be doing the bulk of the work when it comes to crunching numbers for regression in Python.
'''
m = ols('Close ~ Open',df).fit()
print (m.summary())


# In[38]:


m = ols('Close ~ Open + High + Low ',df).fit()
print (m.summary())


# In[ ]:


'''
An example of multivariate linear regression.

In our multivariate regression output above, we learn that by using additional independent variables,
such as the number of High, we can provide a model that fits the data better, 
as the R-squared for this regression has increased to 0.999. 
This means that we went from being able to explain about 99.5% of the variation in the model to 99.9% 
with the addition of a few more independent variables. 
'''


# In[ ]:





# In[37]:


sns.jointplot(x="Close", y="Open", data=df, kind = 'reg',fit_reg= True, size = 7)
plt.show()


# In[45]:


#a k-means cluster model
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster

get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.Low, df.High)
plt.title('Low and High value scatterplot')
plt.xlabel('Low')
plt.ylabel('High')


# In[ ]:


#Step two: Building the cluster model


# In[71]:


#df.drop(df.columns[0:2], axis=1, inplace=True)
df.drop(df.columns[0:2], axis=1, inplace=True)
df.drop(df.columns[6:212], axis=1, inplace=True)
df


# In[72]:


faith = np.array(df)

k = 2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(faith)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[ ]:




