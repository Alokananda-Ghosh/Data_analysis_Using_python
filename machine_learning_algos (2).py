#!/usr/bin/env python
# coding: utf-8

# In[52]:


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


# In[53]:


df.head()


# In[54]:


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


# In[55]:


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


# In[73]:


from sklearn.cluster import KMeans
df.head()
from sklearn.model_selection import train_test_split
kmeans = KMeans(n_clusters = 5)
X = df.values[:, 2:12]
kmeans.fit(X)
df['P.Close'] = kmeans.predict(X)
df.head()
sns.set_context('notebook', font_scale = 1.1)
sns.set_style('ticks')
sns.lmplot('P.Close','Bid', scatter = True, fit_reg = False, data = df, hue = 'Code')


# In[61]:


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[62]:


seed = 7
kfold = KFold(n_splits=10)
cart = DecisionTreeClassifier()


# In[63]:


num_trees = 150


# In[64]:


model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)


# In[65]:


results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[66]:


#The output above shows that we got around 81.6% accuracy of our bagged decision tree classifier model.


# In[14]:


from pandas import read_csv
from sklearn.decomposition import PCA


# In[15]:


pca = PCA(n_components=3)
fit = pca.fit(X)
print(fit.components_)


# In[16]:


from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


# In[17]:


#Higher the score higher the importance of the attribute.
#High and low is very important attribute


# In[18]:


from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[19]:


model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features: %d")
print("Selected Features: %s")
print("Feature Ranking: %s")


# In[20]:


from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer
Data_normalizer = Normalizer(norm='l1').fit(X)
Data_normalized = Data_normalizer.transform(X)


# In[21]:


set_printoptions(precision=2)
print ("\nNormalized data:\n", Data_normalized [0:3])


# In[22]:


#KNN
#i. Training and testing on the entire dataset

x = df.values[:, 4:12]
y = df.values[:, 12]
y=y.astype('int')
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x,y)


# In[23]:


logreg.predict(x)


# In[24]:


y_pred=logreg.predict(x)
len(y_pred)


# In[25]:


from sklearn import metrics
metrics.accuracy_score(y,y_pred)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x,y)


# In[27]:


y_pred=knn.predict(x)
metrics.accuracy_score(y,y_pred)


# In[28]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x,y)


# In[29]:


y_pred=knn.predict(x)
metrics.accuracy_score(y,y_pred)


# In[30]:


#Splitting into train/test
from sklearn.model_selection import train_test_split
x.shape


# In[31]:



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=4)
x_train.shape


# In[32]:


x_test.shape


# In[33]:


y_train.shape


# In[34]:


y_test.shape


# In[35]:


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=knn.predict(x_test)
metrics.accuracy_score(y_test,y_pred)


# In[36]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)


# In[37]:


y_pred=knn.predict(x_test)
metrics.accuracy_score(y_test,y_pred)


# In[38]:


k_range=range(1,15)
scores=[]
for k in k_range:
         knn = KNeighborsClassifier(n_neighbors=k)
         knn.fit(x_train, y_train)
         y_pred = knn.predict(x_test)
         scores.append(metrics.accuracy_score(y_test, y_pred))
scores


# In[39]:


import matplotlib.pyplot as plt
plt.plot(k_range,scores)


# In[40]:


plt.xlabel('k for kNN')


# In[41]:


plt.ylabel('Testing Accuracy')


# In[42]:



plt.show()


# In[43]:


import matplotlib.pyplot as plt
plt.plot(k_range,scores)
plt.xlabel('k for kNN')
plt.ylabel('Testing Accuracy')


# In[44]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
x = df.values[:, 4:10]
y = df.values[:, 4:10]
plt.scatter(x,y)


# In[45]:



kmeans=KMeans(n_clusters=5)
kmeans.fit(x)


# In[46]:


centroids=kmeans.cluster_centers_
labels=kmeans.labels_
centroids


# In[47]:


colors=['g.','r.','c.','y.']
for i in range(len(x)):
         print(x[i],labels[i])
         plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize=10)


# In[48]:


plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=150,linewidths=5,zorder=10)


# In[49]:


plt.show()


# In[50]:


df.plot(kind='density',subplots=True,sharex=False)
plt.show()


# In[51]:


import matplotlib.pyplot as plt
df.hist()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




