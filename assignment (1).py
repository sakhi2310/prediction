#!/usr/bin/env python
# coding: utf-8

# In[89]:


#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix


from sklearn import metrics


warnings.filterwarnings("ignore")

sns.set()


# In[87]:


get_ipython().system('pip install session_info')


# In[90]:


import session_info
session_info.show()


# import dataset(train and test)

# In[2]:


diabetes_df_train1 = pd.read_csv('Downloads\diabetes_v2\diabetes_train_analysis.csv')
diabetes_df_train1.head()


# In[3]:


diabetes_df_test1 = pd.read_csv('Downloads\diabetes_v2\diabetes_test_analysis.csv')
diabetes_df_test1.head()


# In[4]:


diabetes_df_train2 = pd.read_csv('Downloads\diabetes_v2\diabetes_train_info.csv')
diabetes_df_train2.head()


# In[5]:


diabetes_df_test2 = pd.read_csv('Downloads\diabetes_v2\diabetes_test_info.csv')
diabetes_df_test2.head()


# merge the data info and analysis (for both train and test)

# In[6]:


merged_data = diabetes_df_train1.merge(diabetes_df_train2,on=["id"])


# In[7]:


merged_data_test = diabetes_df_test1.merge(diabetes_df_test2,on=["id"])


# In[8]:


merged_data = merged_data.drop('id', axis=1)
merged_data.head()


# In[9]:


merged_data_test = merged_data_test.drop('id', axis=1)

merged_data_test.head()


# count the values to find missing values

# In[10]:


merged_data.count()


# weight have some missing value which cannot be filled manually so we need to drop the column

# In[11]:


merged_data_test.count()


# #Info about the dataset

# In[12]:


merged_data.columns


# In[13]:


merged_data.info()


# In[14]:


merged_data.describe()


# In[15]:


merged_data.isnull()


# In[16]:


merged_data.isnull().sum()


# In[17]:


merged_data_test.isnull().sum()


# In[18]:


df = merged_data.dropna()


# In[19]:


df_t = merged_data_test.dropna()


# In[20]:


df_t


# column "gender" have more than one type of values so we need to replace those and change them into categorical value
# 1 = male
# 2 = female

# In[21]:


df['gender'].value_counts()


# In[22]:


df['gender'] = df['gender'].replace(['male', 'female'], ['m', 'f'])


# In[23]:


df_t['gender'] = df_t['gender'].replace(['male', 'female'], ['m', 'f'])


# In[24]:


df['gender'] = df['gender'].replace(['m', 'f'], ['1', '2'])


# In[25]:


df_t['gender'] = df_t['gender'].replace(['m', 'f'], ['1', '2'])


# In[26]:


df_t


# In[27]:


df['gender'] = df['gender'].astype(int)
df_t['gender'] = df_t['gender'].astype(int)


# in train and test dataset value of age has invalid data to we have to remove that colun as it will affect the accuracy

# In[28]:


df = df.drop('age', axis=1)


# In[29]:


df_t = df_t.drop('age', axis=1)


# ## Converting string into categorical data

# In[30]:


x = {'low':'0', 'medium':'1', 'high':'2'}
df['cholesterol'] = df['cholesterol'].map(x)


# In[31]:


df_t['cholesterol'] = df_t['cholesterol'].map(x)


# In[32]:


df['gluc'] = df['gluc'].map(x)


# In[33]:


df_t['gluc'] = df_t['gluc'].map(x)


# In[34]:


df['gluc'] = df['gluc'].astype(int)


# In[35]:


df_t['gluc'] = df_t['gluc'].astype(int)


# In[36]:


df


# pressure has two type of values "80/120"and"80\120"
# #First we need to make it same and then divide it into two different column

# In[37]:


df2=df.replace(regex=[r'\\'],value='/')


# In[38]:


df2_t=df_t.replace(regex=[r'\\'],value='/')


# In[39]:


df2_t


# In[40]:


print(repr(df.pressure[3]))


# In[41]:


df1 =df2['pressure'].str.split('/',expand =True)


# In[42]:


df1_t =df2_t['pressure'].str.split('/',expand =True)


# In[43]:


df1_t


# In[44]:


df2["systolic"]= df1[0]
 
# making separate last name column from new data frame
df2["diastolic"]= df1[1]


# In[45]:


df2_t["systolic"]= df1_t[0]
 
# making separate last name column from new data frame
df2_t["diastolic"]= df1_t[1]


# In[46]:


df2_t


# Converting string into integer values

# In[47]:


df2['systolic'] = df2['systolic'].astype(int)


# In[48]:


df2_t['systolic'] = df2_t['systolic'].astype(int)


# In[49]:


df2['diastolic'] = df2['diastolic'].astype(int)


# In[50]:


df2_t['diastolic'] = df2_t['diastolic'].astype(int)


# In[51]:


df2['cholesterol'] = df2['cholesterol'].astype(int)


# In[52]:


df_t['cholesterol'] = df_t['cholesterol'].astype(int)


# Drop the column pressure as it is sub divided into systolic and diastolic pressure

# In[53]:


df = df2.drop('pressure', axis=1)


# In[54]:


df_test = df2_t.drop('pressure', axis=1)


# # Data Visualization

# Plotting the data distribution plots

# In[55]:


p = df.hist(figsize = (20,20))


# let’s check that how well our outcome column is balanced

# In[56]:


color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = df["diabetes"].map(lambda x: color_wheel.get(x + 1))
print(df.diabetes.value_counts())
p=df.diabetes.value_counts().plot(kind="bar")


# Here from the above visualization it is clearly visible that our dataset is completely imbalanced in fact the number of patients who are diabetic is half of the patients who are non-diabetic.

# ## Correlation between all the features

# In[57]:


plt.figure(figsize=(12,10))
# seaborn has an easy method to showcase heatmap
p = sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')


# In[58]:


# function to visualize relationship:
def re(colnam1, colnam2):
    plt.figure(figsize=(16, 6))
    sns.regplot(x=df[colnam1], y=df[colnam2])
    sns.set_style("darkgrid")


# In[59]:


re("diabetes", "gluc")


# In[60]:


re("diabetes", "cholesterol")


# # model building

# Splitting the dataset(train and test)

# In[61]:


X_train = df.drop('diabetes', axis=1)
y_train = df['diabetes']


# In[62]:


X_test = df_test.drop('diabetes', axis=1)
y_test = df_test['diabetes']


# ## Random Forest

# Building the model using RandomForest

# In[63]:



rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)


# In[64]:


rfc_train = rfc.predict(X_train)

print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train)))


# Getting the accuracy score for Random Forest

# In[65]:



predictions = rfc.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))


# In[66]:


#Classification report and confusion matrix of random forest model


# In[67]:



print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


# # Decision Tree

# In[68]:



dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# In[69]:




predictions = dtree.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))


# In[70]:


from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


# #  XgBoost classifier

# In[71]:



xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)


# In[72]:



xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test, xgb_pred)))


# In[73]:



print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test,xgb_pred))


# # Support Vector Machine (SVM)

# In[74]:



svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[75]:


svc_pred = svc_model.predict(X_test)


# In[76]:



print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred)))


# In[77]:



print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))


# Accuracy Comparison

# In[78]:


print("Accuracy_Score of random forest =", format(metrics.accuracy_score(y_train, rfc_train)))
print("Accuracy Score svm =", format(metrics.accuracy_score(y_test, svc_pred)))
print("Accuracy Score xgboost =", format(metrics.accuracy_score(y_test, xgb_pred)))
print("Accuracy Score decision tree =", format(metrics.accuracy_score(y_test,predictions)))


# # Feature Importance

# In[79]:


rfc.feature_importances_


# In[80]:


df


# In[81]:


df_test


# In[82]:


(pd.Series(rfc.feature_importances_, index=X_train.columns).plot(kind='barh'))


# # Saving Model – Random Forest and justify the result

# In[83]:




# Firstly we will be using the dump() function to save the model using pickle
saved_model = pickle.dumps(rfc)

# Then we will be loading that saved model
rfc_from_pickle = pickle.loads(saved_model)

# lastly, after loading that model we will use this to make predictions
rfc_from_pickle.predict(X_test)


# In[84]:


df.tail()


# In[85]:


#Putting data points in the model will either return 0 or 1 i.e. person suffering from diabetes or not.
rfc.predict([[1,0,1,0,1,175,85.0,2,152,90]])


# In[ ]:




