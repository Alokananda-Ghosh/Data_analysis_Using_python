#!/usr/bin/env python
# coding: utf-8

# In[325]:


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


# In[326]:


df.loc[df['Close-Open'] <= abs(100), '<= 100'] = 'False'
df.loc[df['Close-Open'] > abs(100), '<= 100'] = 'True'


# In[327]:


df.loc[(df['<= 100'] == 'True') & (df['Time'] == '15:30:00'), 'Status'] = 'Found' 
 
print (df)


# In[328]:


df[df.Status == 'Found']


# In[332]:


df.iat[1,2]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[345]:


df.iat[0,2]


# In[333]:


df


# In[346]:


a=[]
statement=''
for i in range (0,18537):
    if df.iat[i,7]<0:
        if(df.iat[i+1,2]<=df.iat[i,2]):
            print("Profit")
            statement="Profit"
            a.append(statement)
        else:
            print("Loss") 
            statement="Loss"
            a.append(statement)
    elif df.iat[i,7]==0:
        print("None")
        statement="None"
        a.append(statement)
    else:
        if(df.iat[i+1,2]<=df.iat[i,2]):
            print("Loss")
            statement="Loss"
            a.append(statement)
        else:
            print("Profit")
            statement="Profit"
            a.append(statement)
        
        


# In[347]:


a


# In[348]:


df.drop(df.columns[18534:18536], axis=0, inplace=True)
df


# In[ ]:





# In[ ]:





# In[349]:


len(a)


# In[ ]:





# In[350]:


df


# In[ ]:





# In[351]:


df['P'] = a


# In[353]:


df.to_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\NFBNIFC12.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




