#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import pandas as pd
import matplotlib.pyplot  as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[39]:


df=pd.read_csv("https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv")


# In[40]:


df['total_acc'].fillna((df['total_acc'].mean()), inplace=True)
df['longest_credit_length'].fillna(12.0, inplace=True)
df["emp_length"].fillna(0 ,inplace=True)
df['annual_inc'].fillna((df["annual_inc"].mean()), inplace=True)
df['revol_util'].fillna((df['revol_util'].mean()), inplace=True)
df['delinq_2yrs'].fillna(0.0, inplace=True)


# In[41]:


df.drop("addr_state",axis=1,inplace=True)


# In[42]:


df.replace({"36 months":3 ,"60 months" :5 , "MORTGAGE":1,"RENT":2, "OWN":3, "OTHER":4 , "NONE":5 , "ANY":6,"verified" : 0 ,"not verified" :1},inplace=True)


# In[43]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
df['purpose']= label_encoder.fit_transform(df['purpose']) 


# In[44]:


x=df.drop("bad_loan",axis=1)
y=df["bad_loan"]


# In[45]:


from imblearn.over_sampling import RandomOverSampler


# In[46]:


os =  RandomOverSampler(ratio=1)


# In[47]:


X_train_res, y_train_res = os.fit_sample(x, y)


# In[48]:


X_train_res.shape,y_train_res.shape


# In[49]:



from collections import Counter


# In[50]:


print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))


# In[51]:


from sklearn.model_selection import train_test_split
# X=data.drop(["log_loan_amnt","log_int_rate","log_annual_inc","bad_loan"],axis=1)
# Y=data["bad_loan"]
X_train, X_test, Y_train, Y_test = train_test_split(X_train_res, y_train_res, test_size=0.3, random_state=5)


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lreg = LogisticRegression()
lreg.fit(X_train,Y_train)
pred=lreg.predict(X_test)
print(metrics.accuracy_score(Y_test,pred))


# In[53]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = lreg.predict(X_test)

print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[56]:


os_us = SMOTETomek(ratio=0.5)

X_train_res1, y_train_res1 = os_us.fit_sample(x, y)


# In[57]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss


# In[58]:


# Implementing Oversampling for Handling Imbalanced 
smk = SMOTETomek(random_state=42)
X_rest,y_rest=smk.fit_sample(x,y)


# In[59]:


X_rest.shape,y_rest.shape


# In[60]:


from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_rest)))


# In[61]:


X_train, X_test, Y_train, Y_test = train_test_split(X_rest, y_rest, test_size=0.3, random_state=5)


# In[62]:


lg = LogisticRegression()
lg.fit(X_train,Y_train)
pred=lg.predict(X_test)
print(metrics.accuracy_score(Y_test,pred))


# In[66]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = lreg.predict(X_test)

print(confusion_matrix(Y_test, pred))
print(classification_report(Y_test, pred))


# In[63]:


X_train, X_test, Y_train, Y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=5)


# In[64]:


lg = LogisticRegression()
lg.fit(X_train,Y_train)
pred=lg.predict(X_test)
print(metrics.accuracy_score(Y_test,pred))


# In[65]:





# In[ ]:




