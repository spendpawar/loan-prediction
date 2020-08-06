#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd
import matplotlib.pyplot  as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv")


# In[3]:


df['total_acc'].fillna((df['total_acc'].mean()), inplace=True)
df['longest_credit_length'].fillna(12.0, inplace=True)
df["emp_length"].fillna(0 ,inplace=True)
df['annual_inc'].fillna((df["annual_inc"].mean()), inplace=True)
df['revol_util'].fillna((df['revol_util'].mean()), inplace=True)
df['delinq_2yrs'].fillna(0.0, inplace=True)


# In[4]:


df.info()


# In[5]:


df.bad_loan.value_counts()


# In[7]:


df.drop("addr_state",axis=1,inplace=True)


# In[12]:


df.replace({"36 months":3 ,"60 months" :5 , "MORTGAGE":1,"RENT":2, "OWN":3, "OTHER":4 , "NONE":5 , "ANY":6,"verified" : 0 ,"not verified" :1},inplace=True)


# In[13]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
df['purpose']= label_encoder.fit_transform(df['purpose']) 


# In[16]:


x=df.drop("bad_loan",axis=1)
y=df["bad_loan"]


# In[ ]:


from imblearn.under_sampling import NearMiss


# In[17]:


nm = NearMiss(random_state=5)
X_res,y_res=nm.fit_sample(x,y)


# In[15]:


#from imblearn.under_sampling import NearMiss


# In[18]:


X_res.shape,y_res.shape


# In[19]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[25]:


from sklearn.model_selection import train_test_split
# X=data.drop(["log_loan_amnt","log_int_rate","log_annual_inc","bad_loan"],axis=1)
# Y=data["bad_loan"]
X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=5)


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lreg = LogisticRegression()
lreg.fit(X_train,Y_train)
pred=lreg.predict(X_test)
print(metrics.accuracy_score(Y_test,pred))


# In[27]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = lreg.predict(X_test)

print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[ ]:




