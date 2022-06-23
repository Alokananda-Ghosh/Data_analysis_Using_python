#!/usr/bin/env python
# coding: utf-8

# In[156]:


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


# In[157]:


#To get information about null values. 
#If the zero values are low, they must be discarded
print(df.isnull().sum())
#It is necessary to study the data types of data columns.
#Understanding data types ensures that data is collected 
#in the preferred format and the value of each property is as expected.
print(df.dtypes)


# In[158]:


#It is clear that the "Expire Date" column is given as objects instead of date format
df["Date"] = pd.to_datetime(df["Date"], dayfirst = True).dt.date
#Now we need to divide the Expire Date column into columns by years and months. 
#It is importanat step cause entire date is based upon year 2021
#If we divide the date into year and month we can nalyse the change in data on the basis of month
df["Year"]=pd.to_datetime(df["Date"]).dt.year
df["Month"]=pd.to_datetime(df["Date"]).dt.month


# In[159]:


df["Year"]


# In[160]:


df["Month"]


# In[161]:


print(df.groupby("Month")["Close"].mean())
monthmean=df.groupby("Month")["Close"].mean().reset_index()
datamonth=monthmean.set_index("Month")
sns.lineplot(data=datamonth)
plt.xlabel="Month"
plt.ylabel="Close"
plt.title=("Average Close by months")
plt.show()


# In[162]:


print(df.groupby("Date")["Close"].mean())
monthmean=df.groupby("Date")["Close"].mean().reset_index()
datamonth=monthmean.set_index("Date")
sns.lineplot(data=datamonth)
plt.xlabel="Date"
plt.ylabel="Close"
plt.title=("Average Close by date")
plt.show()


# In[163]:


#Binning by distance
min_value = df['Close'].min()
max_value = df['Close'].max()
print(min_value)
print(max_value)


# In[164]:


'''
Now we can calculate the range of each interval, i.e. the minimum and maximum value of each interval.
Since we have 3 groups, we need 4 edges of intervals (bins):

small — (edge1, edge2)
medium — (edge2, edge3)
big — (edge3, edge4)
'''
#We can use the linspace() function of the numpy package to calculate the 4 bins, equally distributed.
import numpy as np
bins = np.linspace(min_value,max_value,4)
bins


# In[165]:


labels = ['small', 'medium', 'big']


# In[166]:


df['bins'] = pd.cut(df['Close'], bins=bins, labels=labels, include_lowest=True)


# In[167]:


import matplotlib.pyplot as plt
plt.hist(df['bins'], bins=3)


# In[168]:


df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
print (df)


# In[169]:


df['hour'].head(20)


# In[170]:


'''
writer = pd.ExcelWriter('Data_Binning2.xlsx') 

df.to_excel(writer, sheet_name='Data_Binning2', header=None, index=False,
         startcol=1, startrow=2)
'''


# In[171]:


'''
with pd.ExcelWriter('Data_Binning2.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet1')
'''


# In[172]:


print(df.groupby("hour")["Close"].mean())
monthmean=df.groupby("hour")["Close"].mean().reset_index()
datamonth=monthmean.set_index("hour")
sns.lineplot(data=datamonth)
plt.xlabel="hour"
plt.ylabel="Close"
plt.title=("Average Close by hour")
plt.show()


# In[173]:


print(df.groupby("hour")["Close-Open"].mean())
monthmean=df.groupby("hour")["Close-Open"].mean().reset_index()
datamonth=monthmean.set_index("hour")
sns.lineplot(data=datamonth)
plt.xlabel="hour"
plt.ylabel="Close-Open"
plt.title=("Average Close-Open by hour")
plt.show()


# In[174]:


df.loc[df['Close-Open'] <= abs(100), '<= 0'] = 'False'
df.loc[df['Close-Open'] > abs(100), '<= 0'] = 'True'


# In[175]:


print(df)


# In[176]:


df.to_csv('Data_Binning.csv', mode='a', index=False, header=False)


# In[177]:



#df.loc[(df['<= 0'] == 'TRUE') | (df['First_name'] == 'Jay'), 'Status'] = 'Found' 
df.loc[(df['<= 0'] == 'TRUE') & (df['Time'] == '3:30:00 PM'), 'Status'] = 'Found' 
 
print (df)


# In[178]:


df[df.Status == 'Found']


# In[179]:


#Binning by Frequency
df['bin_qcut'] = pd.qcut(df['Close'], q=3, precision=1, labels=labels)


# In[180]:


#Sampling
'''
Sampling is another technique of data binning. 
It permits to reduce the number of samples, by grouping similar values or contiguous values.
There are three approaches to perform sampling:
by bin means: each value in a bin is replaced by the mean value of the bin.
by bin median: each bin value is replaced by its bin median value.
by bin boundary: each bin value is replaced by the closest boundary value, i.e. maximum or minimum value of the bin.
'''
'''
In order to perform sampling, the binned_statistic() function of the scipy.stats package can be used. 
This function receives two arrays as input, x_data and y_data,
as well as the statistics to be used (e.g. median or mean) and the number of bins to be created. 
The function returns the values of the bins as well as the edges of each bin.
'''
from scipy.stats import binned_statistic
x_data = np.arange(0,len(df))
#numpy.arange([start, ]stop, [step, ], dtype=None)
y_data = df['Close']
x_bins,bin_edges, misc = binned_statistic(y_data,x_data, statistic="median", bins=len(bins)-1)


# In[181]:


'''
Now we should approximate each value of the df['Close'] column to the median value of the corresponding bin. 
Thus we convert the bin edges to an IntervalIndex, which receives as index the left and right edges of each interval.
In our case, the left edges starts from the beginning of the bin edges and do not contain the last value of the bin edges. 
The right edges instead, start from the second value of the bin edges and last until the last value.
'''
bin_intervals = pd.IntervalIndex.from_arrays(bin_edges[:-1], bin_edges[1:])


# In[182]:


'''
We can quantise the Close column by defining a set_to_median() function 
which loops through the intervals and when it finds the correct interval, 
it returns the mid value.
'''
def set_to_median(x, bin_intervals):
    for interval in bin_intervals:
        if x in interval:
            return interval.mid


# In[183]:


'''
We use the apply() function to apply the set_to_median() to the Close column.
'''
df['sampled_close'] = df['Close'].apply(lambda x: set_to_median(x, bin_intervals))


# In[184]:


#Now we can plot results. We note the loss of information.
plt.plot(df['Close'], label='original')
plt.plot(df['sampled_close'], color='red', label='sampled')
plt.legend()
plt.show()


# In[185]:


'''
Finally, we can plot the median values.
We can calculate the y values (y_bins) corresponding to the binned values (x_bins)
as the values at the center of the bin range.
'''
y_bins = (bin_edges[:-1]+bin_edges[1:])/2
y_bins


# In[186]:


plt.plot(x_data,y_data)
plt.scatter(x_bins, y_bins,  color= 'red',linewidth=5)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




