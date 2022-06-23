#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import re
path = "C://Users//reauter//Desktop//New folder (3)//Book2.csv"
dataset = pd.read_csv(path)
dataset.head()
#Data Preprocessing will be done with the help of following script lines
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:,1].values


# In[4]:


#changing to categorical value
from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

#view transformed values
print(y_transformed)


# In[5]:


#Next, we will divide the data into train and test split. The following code will split the dataset into 70% training data and 30% of testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[6]:


#Next, train the model with the help of RandomForestClassifier class of sklearn as follows
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)


# In[12]:


#At last, we need to make prediction. It can be done with the help of following script −
y_pred = classifier.predict(X_test)
#Next, print the results as follows −
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[11]:


'''
What is the purpose of classification report?
A Classification report is used to measure the quality of predictions
from a classification algorithm. How many predictions are True and how many are False.
More specifically, True Positives, False Positives, True negatives and False Negatives 
are used to predict the metrics of a classification report as shown below.

Precision:- Accuracy of positive predictions.

Precision of code 2102 is 67%.
Precision of code 2103 is 62%.
Precision of code 2104 is 0%.
Precision of code 2106 is 0%.

Recall:- Fraction of positives that were correctly identified.

100% of positives are correctly identified of code 2102
83% of positives are correctly identified of code 2103
0% of positives are correctly identified of code 2104
0% of positives are correctly identified of code 2106


F1 score — What percent of positive predictions were correct

80% of positive predictions were correct of 2102
71% of positive predictions were correct of 2103
0% of positive predictions were correct of 2104
0% of positive predictions were correct of 2106

Support

Support is the number of actual occurrences of the class in the specified dataset.

Code 2102 appeared in the dataset 2
Code 2103 appeared in the dataset 6
Code 2104 appeared in the dataset 2 
Code 2106 appeared in the dataset 1

Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right.
63.63% of predictions our model got right. 
'''


# In[80]:


# importing libraries    
import numpy as nm    
import matplotlib.pyplot as mtp    
import pandas as pd    
datasets = pd.read_csv("C://Users//reauter//Documents//clustering_excel.csv")


# In[82]:


#Extracting Independent Variables −
x = datasets.iloc[:, [0,1]].values


# In[83]:


#Finding the optimal number of clusters using the elbow method
#finding optimal number of clusters using the elbow method  
from sklearn.cluster import KMeans  
wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(x)  
    wcss_list.append(kmeans.inertia_)  
mtp.plot(range(1, 11), wcss_list)  
mtp.title('The Elobw Method Graph')  
mtp.xlabel('Number of clusters(k)')  
mtp.ylabel('wcss_list')  
mtp.show()  


# In[84]:


#The number of cluster will be 3,understood from the above graph
#training the K-means model on a dataset  
kmeans = KMeans(n_clusters=2, init='k-means++', random_state= 42)  
y_predict= kmeans.fit_predict(x)  


# In[87]:


#visulaizing the clusters  
mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
#mtp.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
#mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
#mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
mtp.title('Clusters of Shares')  
mtp.xlabel('Code')  
mtp.ylabel('P.close')  
mtp.legend()  
mtp.show()  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




