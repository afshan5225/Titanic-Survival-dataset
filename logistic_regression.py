# -*- coding: utf-8 -*-
"""Logistic Regression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vakwzFRSWDfSlTCwPFhufaTGNDQyRZdm

#**TITANIC SURVIVAL DATASET**

importing libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix

"""importing datset

"""

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df.head()

df.tail()

df.shape

df.info()

df.isna().sum()

#percentage of null in age col

(df.Age.isna().sum()/len(df.Age))*100

#% of null in cabin
(df.Cabin.isna().sum()/len(df.Cabin))*100

df.Cabin.unique()

#re = df.groupby(['Survived','Cabin'])['Cabin']
#df['column_name'].str.extract(r'([A-Za-z])')

df.drop(columns = ['Cabin'],inplace =True)

df.head()

df.Embarked.unique()

df[df['Fare'] >= 70][['Embarked', 'Fare']]
#practised

#show embarked values were Pclass are 1
df[df['Pclass']==1][['Embarked','Pclass']].value_counts()

#show embarked values were Pclass are 2
df[df['Pclass']==2][['Embarked','Pclass']].value_counts()

#df[(df['Fare'] >= 70) & (df['Fare'] <= 100)][['Embarked', 'Fare']]
#practise

#since Embarked failed to show any relation, we are droping it

#df.drop(columns =['Embarked'],inplace =True)

df.head()

"""#fixing age"""

#mean of age
a = np.mean(df['Age'])
a

df.Age.median() #median - more suitable since wont there be much fuluctuations

df.Age.mode() #mode

#box plot

df['Age'].plot.box()

sns.boxplot(df.Age)

df.Age.value_counts()

df['Age'].fillna(df.Age.median(),inplace =True)

df.Age.isna().sum()

"""#drop nan embarked"""

df.dropna(inplace =True)

df.isna().sum()

df.shape

df.head()

#dropping passenger ID

df.drop(columns =['PassengerId','Name','Ticket'],inplace =True)

df.head()

df['Sex'].value_counts().plot(kind='bar')
plt.grid()
plt.title("M vs W")

df.groupby('Sex')['Survived'].value_counts().plot(kind ='bar')

plt.grid()
plt.title('survival rate')

#ploting a graph to findout the strength of p class

df.Pclass.value_counts().plot(kind = 'bar')

plt.title("P class")

#pClass survival rate

df.groupby('Survived')['Pclass'].value_counts().plot(kind ='bar')

#or
sns.countplot(x ='Survived', data =df, hue ='Pclass')
plt.grid()

"""#Encoders"""

from sklearn.preprocessing import LabelEncoder

L = LabelEncoder()

df.Sex = L.fit_transform(df.Sex)

df.Embarked = L.fit_transform(df.Embarked)

#converting dtype of age to int

df.Age = df.Age.astype(float).astype(int)

df.info()

df.head()

df.Fare = round(df.Fare,2)

df.head()

sns.heatmap(df.corr(),annot= True)

sns.pairplot(df)

"""Feature importance"""

df.head()

X = df.iloc[:,1:]
y =df.iloc[:,0]

from sklearn.ensemble import ExtraTreesClassifier

feat = ExtraTreesClassifier()

feat.fit(X,y)

feat.feature_importances_

feat_imp = pd.Series(feat.feature_importances_, index =X.columns)
feat_imp.nlargest(5).plot(kind ='bar')

"""Spliting the data"""

skf = StratifiedKFold(n_splits=5)

for train_index,test_index in skf.split(X,y):
  X_train,X_test = X.iloc[train_index],X.iloc[test_index]
  y_train,y_test = y.iloc[train_index],y.iloc[test_index]

"""Model Selection"""

classifier = LogisticRegression()

"""Training the model"""

classifier.fit(X_train,y_train)

"""Model testing"""

y_pred = classifier.predict(X_test)

"""EDA"""

final_df = pd.DataFrame({"Actual":y_test,"Pred":y_pred})
final_df

sns.heatmap(data = final_df.corr(),annot =True)

"""performance matrix"""

#confusion matrix


confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report,accuracy_score

accuracy_score(y_test,y_pred)

classification_report(y_test,y_pred)

"""Exportation of model (dumbing)
module - pickle
"""

import pickle

picks = pickle.dumps(classifier)