#!/usr/bin/env python
# coding: utf-8

# In[62]:


#Starting implementation
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
df = pd.read_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\Currency working(Cleaned data).csv")
#df.columns = ["Full_Code","Name", "Code","L.Trade", "Bid Vol","Bid","Ask","Ask Vol","Volume","High","Low","P.Close","+/- %"]
df.head()

#implementation
from sklearn.model_selection import train_test_split
decision = tree.DecisionTreeClassifier(criterion="gini")
X = df.values[:, 4:12]
Y = df.values[:, 12]
Y=Y.astype('int')
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
decision.fit(trainX, trainY)
print("Accuracy: \n", decision.score(testX, testY))


# In[63]:


df.head()


# In[89]:


from sklearn import svm
df = pd.read_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\Currency working(Cleaned data).csv")
df.columns = ["Full_Code","Name", "Code","L.Trade", "Bid Vol","Bid","Ask","Ask Vol","Volume","High","Low","P.Close","+/- %"]
from sklearn.model_selection import train_test_split
support = svm.SVC()
x = df.values[:, 4:12]
y = df.values[:, 12]
y=y.astype('int')
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')
sns.lmplot('Code','Bid', scatter=True, fit_reg=False, data=df, hue="Name",col="Name", height=6, aspect=.3, x_jitter=.1)
plt.ylabel('Bid')
plt.xlabel('Code')


# In[77]:


from sklearn import svm
df = pd.read_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\Currency working(Cleaned data).csv")
df.columns = ["Full_Code","Name", "Code","L.Trade", "Bid Vol","Bid","Ask","Ask Vol","Volume","High","Low","P.Close","+/- %"]
from sklearn.model_selection import train_test_split
support = svm.SVC()
x = df.values[:, 4:12]
y = df.values[:, 12]
y=y.astype('int')
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')
sns.lmplot('P.Close','Bid', scatter=True, fit_reg=False, data=df, hue="Code", col="Code", height=6, aspect=.4, x_jitter=.1)
plt.ylabel('Bid')
plt.xlabel('P.Close')


# In[93]:


g = sns.lmplot(x="Code", y="Volume", row="Code", col="L.Trade",
               data=df, height=3)
plt.ylabel('Volume')
plt.xlabel('Code')


# In[ ]:




