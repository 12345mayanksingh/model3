#!/usr/bin/env python
# coding: utf-8

# In[118]:


#This is the problem Statement no-2(Titanic problem) 


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Titanic/titanic_train.csv')
data.head()


# In[3]:


data.sample(n=10)


# In[4]:


data.columns


# In[5]:


data.shape


# In[ ]:


#In this data there are 891 rows and 12 columns are present in which one is target variable(label)
#By studying the data we clearly understand that it is a classfication based problem in which we have to predict whether the 
#person survived or not.


# In[6]:


data.info()


# In[7]:


data.dtypes


# In[ ]:


#we have checked their data types.


# In[8]:


data.isnull().sum()


# In[10]:


#There are nulls present in 'Age','Cabin'and 'embarked'
#so we will treat nulls first of age and embarked but in cabin there are so much nulls so we will drop the column.


# In[14]:


data['Age'].fillna(data.groupby('Sex')['Age'].transform('mean'),inplace=True)
data


# In[15]:


data.isnull().sum()


# In[ ]:


#we have treated the nulls of 'Age'by their average of male and female.


# In[16]:


data.drop('Cabin',axis=1,inplace=True)


# In[40]:


for i in data.columns:
    print(data[i].value_counts())


# In[17]:


data.head()


# In[18]:


data['Embarked'].fillna(value=data['Embarked'].mode()[0],inplace=True)


# In[19]:


data.sample(n=10)


# In[20]:


data.isnull().sum()


# In[21]:


#All the nulls have now been treated.
#we will drop 'PassengerId column 'also it is of no use


# In[22]:


data.drop('PassengerId',axis=1,inplace=True)


# In[23]:


data.head()


# In[24]:


data.describe()


# In[ ]:


#There is a little bit skewness is present in case of 'Fare' because there is differnece between mean and median 
#The difference between 75% and max is high there might be a outliers.
#Now we will drop name and ticket column because they are of no use.


# In[25]:


data.drop(['Name','Ticket'],axis=1,inplace=True)


# In[26]:


data.head()


# In[ ]:


#Now we replace encode the 'Sex'and 'Embarked'features


# In[28]:


data['Sex'].replace({'male':1,'female':0},inplace=True)


# In[29]:


data


# In[31]:


data['Embarked'].replace({'S':0,'C':1,'Q':2},inplace=True)


# In[32]:


data


# In[36]:


plt.figure(figsize=(15,20),facecolor='Red')
plotnumber=1
for colu in data:
    if plotnumber<=9:
        ax=plt.subplot(4,4,plotnumber)
        sns.distplot(data[colu])
        plt.xlabel(colu,fontsize=10)
    plotnumber+=1
plt.show()    


# In[ ]:


#after studying the graph there is a skewness present in 'SibSp','Parch','Fare' columns.
#Now we will check for otliers by plotting the boxplot
#we will check for skewness also


# In[37]:


data.skew()


# In[ ]:


#now we will remove skewness of Fare because rest are categorical data.


# In[44]:


data['Fare']=np.cbrt(data['Fare'])


# In[45]:


data.skew()


# In[ ]:


#now the skewness has been treated now we will check for outliers.


# In[46]:


plt.figure(figsize=(15,20), facecolor="Red")
plotnumber=1
for col in data:
    if plotnumber<=9:
        ax=plt.subplot(4,4,plotnumber)
        sns.boxplot(data[col])
        plt.xlabel(col,fontsize=10)
    plotnumber+=1 
plt.show()    
    


# In[47]:


#There are outliers present in the data now we will remove outliers of 'Fare' features.


# In[48]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
iqr=q3-q1


# In[50]:


Fare_high=q3['Fare']+1.5*iqr['Fare']
Fare_high


# In[51]:


np_index=np.where(data['Fare']>Fare_high)
np_index


# In[52]:


data=data.drop(data.index[np_index])
data.shape


# In[53]:


data.reset_index()


# In[54]:


Fare_low=q1['Fare']-1.5*iqr['Fare']
Fare_low


# In[55]:


np_inde=np.where(data['Fare']<Fare_low)
np_inde


# In[56]:


data=data.drop(data.index[np_inde])
data.shape


# In[57]:


data.reset_index()


# In[58]:


#the ouliers are removed


# In[59]:


corr=data.corr()
corr


# In[ ]:


#now we will check for Vif values


# In[61]:


x=data.drop('Survived',axis=1)
y=data['Survived']



# In[65]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,classification_report
from sklearn.linear_model import LogisticRegression


# In[68]:


Scalar=StandardScaler()
Xscaled=Scalar.fit_transform(x)
Xscaled.shape[1]


# In[69]:


vif=pd.DataFrame()
vif['vifvalues']=[variance_inflation_factor(Xscaled,i) for i in range(Xscaled.shape[1])]
vif['features']=x.columns


# In[70]:


vif


# In[ ]:


#All the vif values are less than 5 so it means that there is not multicollinearity.


# In[ ]:


#All the preprocessing steps are completed now model building.


# In[71]:


x_train,x_test,y_train,y_test=train_test_split(Xscaled,y,test_size=0.25,random_state=200)


# In[98]:


def metric_score(clf,x_train,x_test,y_train,y_test,train=True):
    if train:
        y_pred=clf.predict(x_train)
        print(f"Accuracy score:{accuracy_score(y_train,y_pred)*100:.2f}%")
        
    elif train==False:
        pred=clf.predict(x_test)
        print(f"Accuracy score:{accuracy_score(y_test,pred)*100:.2f}%")
        print(classification_report(y_test,pred,digits=2))
        


# In[99]:


LR=LogisticRegression()
LR.fit(x_train,y_train)


# In[101]:


metric_score(LR,x_train,x_test,y_train,y_test,train=True)

metric_score(LR,x_train,x_test,y_train,y_test,train=False)


# In[102]:


#Train result=79.75%
#test result=81.78%


# In[ ]:


#Now we will build another model -Knn


# In[103]:


from sklearn.neighbors import KNeighborsClassifier


# In[106]:


Knn=KNeighborsClassifier()
Knn.fit(x_train,y_train)


# In[107]:


metric_score(Knn,x_train,x_test,y_train,y_test,train=True)

metric_score(Knn,x_train,x_test,y_train,y_test,train=False)


# In[108]:


#train result=85.51%
#Test result=83.64%


# In[109]:


#NOW we will try to increase the accuracy by GridSearchCV


# In[110]:


from sklearn.model_selection import GridSearchCV


# In[112]:


param_grid={'algorithm':['Kd tree','brute'],
           'leaf_size':[3,5,6,7,8],
           'n_neighbors':[3,5,7,9,11,13]}


# In[113]:


gridsearch=GridSearchCV(estimator=Knn,param_grid=param_grid)
gridsearch.fit(x_train,y_train)


# In[114]:


gridsearch.best_score_


# In[115]:


gridsearch.best_estimator_


# In[ ]:


#our result is accurate no change maximum accuracy is achieved


# In[116]:


import pickle
filename='data_LR.pkl'
pickle.dump(LR, open(filename,'wb'))


# In[125]:


#Problem statement no-1


# In[136]:


df=pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/World%20Happiness/happiness_score_dataset.csv')
df.head()


# In[137]:


df.columns


# In[138]:


df['Region'].value_counts()


# In[139]:


df.shape


# In[140]:


df.describe()


# In[141]:


df.isnull().sum()


# In[142]:


df.drop(["Country", "Region", "Happiness Rank"], axis=1, inplace=True)


# In[143]:


df.head()


# In[147]:


plt.figure(figsize=(20,15),facecolor='red')
plotnumber=1
for column in df:
    if plotnumber<=8:
        ax=plt.subplot(2,4,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column, fontsize=20)
        
    plotnumber+=1
plt.tight_layout()


# In[148]:


y=df['Happiness Score']
x=df.drop(columns=['Happiness Score'])


# In[149]:


plt.figure(figsize=(15,10), facecolor='yellow')
plotnumber=1
for column in x:
    if plotnumber<=8:
        ax=plt.subplot(2,4, plotnumber)
        plt.scatter(x[column], y)
        plt.xlabel(column, fontsize=10)
        plt.ylabel('Happiness score', fontsize=10)
    plotnumber+=1
plt.tight_layout()


# In[150]:


scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)


# In[151]:


x_train, x_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.25, random_state=112)


# In[153]:


from sklearn.linear_model import LinearRegression


# In[155]:


regression=LinearRegression()


# In[156]:


regression.fit(x_train, y_train)


# In[157]:


regression.score(x_train, y_train)


# In[158]:


regression.score(x_test, y_test)


# In[159]:


x_test[1]


# In[160]:


y_pred=regression.predict(x_test)
y_pred


# In[161]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual happiness score')
plt.ylabel('predicted happiness score')
plt.title('actual vs predicted')
plt.show()


# In[ ]:


#99% accuracy achieved no need for hyperparameter tuning


# In[162]:


import pickle
filename='data_LR.pkl'
pickle.dump(LR, open(filename,'wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




