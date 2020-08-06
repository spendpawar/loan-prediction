#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot  as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import seaborn as sns


# In[6]:


#reading dataset
df=pd.read_csv("loan.csv")


# In[7]:


#Displating top 5 records

df.head(5)


# In[8]:


label="Good Customers","Bad customers"
df["bad_loan"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, labels=label ,
                                              fontsize=12, startangle=70)
plt.savefig('bad_loan_pie.png')


# In[9]:


label="verified","Not verified"
df["verification_status"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, labels=label ,
                                              fontsize=12, startangle=30)
plt.savefig('verif_pie.png')


# In[10]:


#finding null values
df.isna().sum()


# In[11]:


#categorical feature present in dataset
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']


# In[12]:


categorical_features


# In[13]:


#checking unique value within categorical feature

for feature in categorical_features:
    print(df[feature].unique())


# In[14]:


continous_features=[feature for feature in df.columns if df[feature].dtype!='O']


# In[15]:


continous_features


# In[16]:


#shows the count of purpose for loan
#high for debt_conslidation and very low for renewable energy

plt.figure(figsize=(15,8))
sns.countplot(y="purpose", hue="bad_loan", data=df , palette="Set1")
plt.savefig('purpose.png')


# In[17]:


#count is high for 36 months

#plt.figure(figsize=(10,8))
sns.countplot(y="term", hue="bad_loan", data=df , palette="Set1")
plt.savefig('term.png')


# In[18]:


sns.countplot("bad_loan", data=df)
plt.savefig('bad_loan.png')


# In[19]:


sns.countplot(x="verification_status",hue="bad_loan",data=df )


# In[20]:


plt.figure(figsize=(20,50))
sns.countplot(y="addr_state",hue="bad_loan" , data=df)
plt.savefig('addr_state.png')


# In[21]:


sns.countplot(x="home_ownership", hue="bad_loan", data=df)


# In[22]:


sns.countplot(x="emp_length", hue="bad_loan", data=df)


# In[23]:


g=sns.distplot(df["loan_amnt"])
g.set_xlabel("Loan Amount Value", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)
g.set_title("Loan Amount Distribuition", fontsize=20)

plt.savefig('dist_loan.png')


# In[24]:


#sns.distplot(data["annual_inc"])


# In[25]:


g=sns.distplot(df["int_rate"])
g.set_xlabel("int_rate Value", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)
g.set_title("intrest rate Distribuition", fontsize=15)

plt.savefig('dist_intrate.png')


# In[26]:


sns.boxplot(x="verification_status", y="loan_amnt", data=df)


# In[27]:


df.drop("bad_loan" , axis=1).plot(kind="box",subplots=True , layout=(3,3) , figsize=(8,8))


# In[28]:


corrmat = df.corr()


# In[29]:


corrmat


# In[30]:


#correlation between column
# from graph it shows that column are not highly correlated with each other ie.they are equaly important.

f, ax = plt.subplots(figsize =(15, 8)) 
plot=sns.heatmap(corrmat, vmax=1, square=True,center=0 , linewidths=0.2, annot=True)
plt.savefig('heatmap.png')


# missing value

# In[31]:


data=df.copy()


# In[32]:


data.isna().sum()


# In[33]:


#emplength
data["emp_length"].fillna(0 ,inplace=True)


# In[34]:


#after replacing Nan value std shows very little change.
print(df.emp_length.std())
print(data.emp_length.std())


# In[35]:


df['longest_credit_length'].value_counts()


# In[36]:


df['longest_credit_length'].mode()


# In[37]:


data['delinq_2yrs'].fillna(0.0 , inplace =True)


# In[38]:


data["revol_util"].mean()


# In[39]:


data["revol_util"].fillna(54.07 , inplace =True)


# In[40]:


print(df.revol_util.std())
print(data.revol_util.std())


# In[41]:


data['total_acc'].fillna((data['total_acc'].mean()), inplace=True)


# In[42]:


data['longest_credit_length'].mode()


# In[43]:


data['longest_credit_length'].fillna(12.0, inplace=True)


# In[44]:


data['annual_inc'].fillna((df["annual_inc"].mean()), inplace=True)


# In[45]:


categorical_features


# In[46]:


clean={"term ": {"36 months":3 ,"60 months" :5 } , 
"home_ownership" :{"MORTGAGE":1,"RENT":2, "OWN":3, "OTHER":4 , "NONE":5 , "ANY":6},
"purpose":{"verified" : 0 ,"not verified" :1}}


# In[47]:


data.replace({"36 months":3 ,"60 months" :5 , "MORTGAGE":1,"RENT":2, "OWN":3, "OTHER":4 , "NONE":5 , "ANY":6,"verified" : 0 ,"not verified" :1},inplace=True)


# In[48]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
data['purpose']= label_encoder.fit_transform(data['purpose']) 


# In[49]:


data['addr_state']= label_encoder.fit_transform(data['addr_state']) 


# In[50]:


data.head(3)


# In[51]:


#data.drop("addr_state" , axis=1 , inplace=True)


# In[52]:


import scipy.stats as stat
import pylab


# In[53]:


#### If you want to check whether feature is guassian or normal distributed
#### Q-Q plot
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()


# In[ ]:





# In[54]:


data['loan_amnt']=data.loan_amnt**(1/2)
plot_data(data,'loan_amnt')


# In[55]:


plot_data(data,'int_rate')


# In[56]:


data['int_rate']=data.int_rate**(1/2)
plot_data(data,'int_rate')


# In[57]:


data['annual_inc']=data.annual_inc**(1/2)
plot_data(data,'annual_inc')


# In[58]:


#data['log_loan_amnt'] = np.log(data['loan_amnt'])


# In[59]:


#data['log_int_rate'] = np.log(data['int_rate'])


# In[60]:


#data['log_annual_inc'] = np.log(data['annual_inc'])


# In[61]:


data.head()


# In[62]:


from sklearn.model_selection import train_test_split
x=data.drop(["bad_loan"],axis=1)
y=data["bad_loan"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)


# In[ ]:





# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lreg = LogisticRegression()
lreg.fit(x_train,y_train)
pred=lreg.predict(x_test)
print(metrics.accuracy_score(y_test,pred))


# In[64]:


from sklearn import tree
from sklearn.model_selection import cross_val_score
Scores=cross_val_score(lreg,x,y, cv=10)

print(Scores)

avg=np.mean(Scores)
print("avg accuracy is:",np.mean(Scores))


# In[ ]:





# In[ ]:





# In[65]:


while True:
    i = input("Enter the value seperated by comma: ")
    if i == "q":
        break
    l = [float(x) for x in i.split(",")]
    sample = np.array(l)
    sample = sample.reshape(1,-1)
#     sample = np.array([6.1, 3.9, 5.1, 1.98])

    print ("loan defaulter {} ".format(lreg.predict(sample)))


# In[ ]:





# In[ ]:





# In[68]:


# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# pred = lreg.predict(x_test)

# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred))


# In[ ]:


data.head(3)


# In[ ]:





# In[ ]:


# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# scaler.fit(X)
# z=scaler.transform(X)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




