#!/usr/bin/env python
# coding: utf-8

# In[414]:


import pandas as pd


# In[415]:


import seaborn as sns


# In[416]:


import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


# In[417]:


import numpy as np
import re


# In[418]:


data=pd.read_csv("C:\\Users\\reauter\\Desktop\\New folder (3)\\Book1.csv")
data.drop("Name_Total",axis=1,inplace=True)


# In[419]:


print(data.head())


# In[420]:


#the number of rows and columns
print(data.shape )


# In[421]:


#Check the duplicate value
print(data.duplicated().sum())


# In[422]:


#To get information about null values. 
#If the zero values are low, they must be discarded
print(data.isnull().sum())
#It is necessary to study the data types of data columns.
#Understanding data types ensures that data is collected 
#in the preferred format and the value of each property is as expected.
print(data.dtypes)


# In[423]:


#It is clear that the "Expire Date" column is given as objects instead of date format
data["Expire Date"] = pd.to_datetime(data["Expire Date"], dayfirst = True).dt.date
#Now we need to divide the Expire Date column into columns by years and months. 
#It is importanat step cause entire date is based upon year 2021
#If we divide the date into year and month we can nalyse the change in data on the basis of month
data["Year"]=pd.to_datetime(data["Expire Date"]).dt.year
data["Month"]=pd.to_datetime(data["Expire Date"]).dt.month


# In[424]:


#It is necessary to study the total L.Trade over the months and put it into an understandable form
data.groupby("Month")["L.Trade"].sum()


# In[425]:


plt.figure(figsize=(10,6))
sns.lineplot(data=data.groupby("P.Close")["Bid Vol"].sum(),label="Bid Vol")
plt.xlabel="P.Close "
plt.ylabel="Bid Vol"
plt.title=("Bid Vol Quantity By P.Close")
plt.show()
#Interpretation
'''
Previous close almost always refers to the prior day's final price
of a security when the market officially closes for the day.
When the Previous close is 0 then the Bid Volume is the highest.
When the previous close is between 0-10000 then the Bid Volume seems to be second highest.

'''


# In[426]:


data["Year"]


# In[427]:


data["Month"]


# In[428]:


#It is necessary to study the total L.Trade over the months and put it into an understandable form
data.groupby("Month")["L.Trade"].sum()
plt.figure(figsize=(10,6))
sns.lineplot(data=data.groupby("Month")["L.Trade"].sum(),label="L.Trade")
plt.xlabel="Month"
plt.ylabel="L.Trade"
plt.title=("L.Trade Quantity By Month")
plt.show()


# In[429]:


#It is necessary to study the average O.
#Price for months and put it into an understandable form
print(data.groupby("Month")["O.Price"].mean())
monthmean=data.groupby("Month")["O.Price"].mean().reset_index()
datamonth=monthmean.set_index("Month")
sns.lineplot(data=datamonth)
plt.xlabel="Month"
plt.ylabel="O.Price"
plt.title=("Average O.Price by months")
plt.show()


# In[430]:


#Interpretation:
#On the third month the O.Price is the lowest 
#but on the 5th month the O.Price is at the peak though it gradually decreses after 5th month.


# In[431]:


#
print(data.groupby("Ask Vol")['O.Price'].sum().sort_values(ascending=False))
datacategory=data.groupby("Ask Vol")['O.Price'].sum().sort_values(ascending=False).reset_index().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="O.Price",y="Ask Vol",data=datacategory).set(title="O.Price by Ask Vol")
plt.show()
#Interpretation
'''
Ask volume is the lowest for the O.Price 334502.25.The Ask Volume is the highest for the O.Price 4590.0.
'''


# In[432]:


print(data.groupby("Name")['O.Price'].sum().sort_values(ascending=False))
datacategory=data.groupby("Name")['O.Price'].sum().sort_values(ascending=False).reset_index()  
plt.figure(figsize=(10,6))
sns.barplot(x="O.Price",y="Name",data=datacategory).set(title="O.Price by Name")
plt.show()
#SILVERMICRO 2104 has the highest O.Price in trade.
#CARDAMOM 2103 and CARDAMOM 2102 have the lowest O.Price


# In[433]:


#Name by Volume
print(data.groupby("Name")['Volume'].sum().sort_values(ascending=False).reset_index().head(10))
datasub=data.groupby("Name")['Volume'].sum().sort_values(ascending=False).reset_index().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="Volume",y="Name",data=datasub).set(title="Volume By Name")
plt.show()
#Interpreation:
#NATURALGAS has the highest Volume in trade
#GOLD-M 2103 has the lowest volume in trade.


# In[434]:


#+/- % by Name
print(data.groupby("Name")['+/- %'].sum().sort_values(ascending=False).reset_index().head(10))
datasub=data.groupby("Name")['+/- %'].sum().sort_values(ascending=False).reset_index().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="+/- %",y="Name",data=datasub).set(title="+/- % By Name")
plt.show()
#MCXMETAL 2102 has the maximum +/- % and MENTHAOIL 2103 has the lowest +/- %.


# In[435]:


#+/- % fluctuation is the highest on 2021-02-17 and lowest on 2021-03-24
data["Expire Date"] = pd.to_datetime(data["Expire Date"]).dt.date
print(data.groupby("Expire Date")["+/- %"].sum())

datatend=data.groupby("Expire Date")["+/- %"].sum()
plt.figure(figsize=(20,5))
sns.lineplot(data=datatend).set(title="+/- % Trends By Month")
plt.ylabel=("+/- %")


# In[436]:


datatend=data.groupby("Time")["+/- %"].sum()
plt.figure(figsize=(20,5))
sns.lineplot(data=datatend).set(title="+/- % Trends By Time")
plt.ylabel=("+/- %")
#Interpretation:
#The +/- % is the lowest at 18:10:34 and +/- % is the highest at 18:11:07.


# In[437]:


print(data.groupby("Name")['Bid'].sum().sort_values(ascending=False))
datacategory=data.groupby("Name")['Bid'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x="Bid",y="Name",data=datacategory).set(title="Bid by Name")
plt.show()
#Interpretation:
#SILVER-M 2104 has the highest Bid which is 69603.00 
#and CARDAMOM 2103 has the lowest Bid.


# In[438]:


print(data.groupby("Bid")['Bid Vol'].sum().sort_values(ascending=False))
datacategory=data.groupby("Bid")['Bid Vol'].sum().sort_values(ascending=False).reset_index().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="Bid Vol",y="Bid",data=datacategory).set(title="Bid Vol by Bid")
plt.show()
#Bid is the highest when the bid volume is is 5.
#Bid is the lowest when the bid volume is 12 and 154 


# In[439]:


corr=data.corr()
import matplotlib.pyplot as plt
import seaborn as sb
corr=data.corr()
sb.heatmap(corr,annot=True)
plt.show
#The correlation between Bid Volume and Volume is the highest.
#which means if the bid volume changes by 1% then volume will also change by 0.969%.


# In[440]:


groupby = data.groupby('code', axis=0)
  
groupby.mean()

#On the basis of average code the other values are plotted


# In[441]:


data['Name'].is_unique


# In[442]:


data['code'].is_unique


# In[481]:


name_code = pd.DataFrame(data,columns=['Name','Code'])


# In[484]:


name_code.head(57)


# In[445]:


name=data['Name']


# In[446]:


gold=name.str.contains('GOLD')


# In[447]:


gold


# In[485]:


data.head(57)


# In[449]:


name_code


# In[450]:


name_code.plot(x='Name',y='code',kind='hist')


# In[534]:



df2=(data[data.Ask<34911.925])

df2
df3 = pd.DataFrame(df2,columns=['code','O.Int','Ask','Name'])


# In[ ]:





# In[535]:


df3.plot(x='Name',y='Ask',kind='line',label="Filtered Ask value less than 34911.925")


# In[536]:


type(data.Name)


# In[545]:


data['Bid'] = data['Bid'].astype(float)
df4=(data[data.Volume>100])
df5 = pd.DataFrame(df4,columns=['L.Trade','Expire Date','Name'])


# In[ ]:





# In[546]:


combined_df = df3.merge(df5, on='Name',how='inner')


# In[560]:



combined_df


# In[561]:


combined_df.plot(x='Name',y='O.Int',kind='line')


# In[562]:


new_df=pd.concat([name_code,combined_df],axis=1)


# In[521]:


new_df


# In[567]:


new_df2=pd.concat([df2,df5],axis=1)


# In[568]:


new_df2


# In[610]:


new_df2.plot(x='Name',y='Volume',kind='line',label="Filtered Ask value less than 34911.925")


# In[611]:


new_df2=data[data.High<35177.075]


# In[580]:


new_df2


# In[581]:


datacat=new_df2.groupby("Name")['O.Price'].sum().sort_values(ascending=False).reset_index()  
plt.figure(figsize=(10,6))
sns.barplot(x="O.Price",y="Name",data=datacat).set(title="O.Price by Name for high<35177.075")
plt.show()


# In[587]:





# In[583]:





# In[589]:


datacata=new_df2.groupby("Ask Vol")['O.Price'].sum().sort_values(ascending=True).reset_index()  
plt.figure(figsize=(10,6))
sns.barplot(x="O.Price",y="Ask Vol",data=datacata).set(title="O.Price by Name for high<35177.075")
plt.show()


# In[590]:


mentions_fed = data["Name"].str.contains("GOLD")
type(mentions_fed)
mentions_fed


# In[594]:


mentions_fed.groupby(data["L.Trade"], sort=False).sum()


# In[600]:


mentions_fed


# In[ ]:





# In[ ]:




