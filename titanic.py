# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:49:56 2019

@author: rahul
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset=pd.read_csv('train.csv')
dataset_test=pd.read_csv('test.csv')
dataset.describe()
dataset.isnull()
dataset.info()
"""visualization"""
sns.catplot(x='Sex',data=dataset,kind="count",hue='Pclass')
sns.catplot(x='Pclass',data=dataset,kind="count",hue='Sex')
dataset['Age'].isnull().sum()#177 null values, we need to fill it
"""here you can either drop entire null age row or fill mean value, the later one is good as 
    you don not lose other columns value"""
dataset['Age']=dataset['Age'].fillna(dataset.Age.mean())#or you can use imputer function
dataset['Age'].isnull().sum()


sns.catplot(x='Embarked',data=dataset,kind="count",hue='Pclass') 
sns.catplot(x='Survived',y='Age',data=dataset,hue='Sex',kind="point") 
print(dataset[dataset.Survived == 0].groupby('Sex')['Age'].count())#female survived more
print(dataset[dataset.Survived == 0].groupby('Pclass')['Age'].count())#class 1 survived more
dataset[['Pclass', 'Survived']].groupby(['Pclass']).mean() #if you want mean instead of count, makes job easier
print(dataset[dataset.Survived == 0].groupby('Age')['Sex'].count())#female survived more
dataset[['SibSp', 'Survived']].groupby(['SibSp']).mean() #person with 1 sibling/spouse survived most
dataset[['Parch', 'Survived']].groupby(['Parch']).mean() #with 3,survived more,also check count of each  
g = sns.FacetGrid(dataset, col='Survived')
g.map(plt.hist, 'Age', bins=20)
sns.catplot(x='Age',data=dataset,kind="count",hue='Survived')#this can also be use instead of above line to get info we neeed
 sns.catplot(x='Age',y='Pclass',data=dataset,kind="point",hue='Survived')
 """this can give you right info sometimes but it is too
 complicated,the best way here is to use hist command"""
 grid = sns.FacetGrid(dataset, col='Survived', row='Pclass', size=2.2, aspect=1.6)#https://seaborn.pydata.org/generated/seaborn.FacetGrid.html#seaborn.FacetGrid
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



freq_port = dataset.Embarked.dropna().mode()[0]


"""simple way to use classification"""
dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

dataset_test.Sex.isnull().sum()
dataset_test['Sex'] = dataset_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


dataset_test['Embarked'] = dataset_test['Embarked'].fillna(freq_port)

dataset_test['Embarked'] = dataset_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

"""dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())"""
dataset_test['Age']=dataset_test['Age'].fillna(dataset_test['Age'].mean())

x=dataset.iloc[:,[2,4,5,6,7,11]]
x_test=dataset_test.iloc[:,[1,3,4,5,6,10]]
y=dataset.iloc[:,1]
"""here you can split dataset in to training and testing set as we don't have test file output,for making classification"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))

names = []
scores = []
for name, model in models:
    model.fit(x,y)
    y_pred = model.predict(x_test)
    scores.append(model.score(x,y))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

submission = pd.DataFrame({
        "PassengerId": dataset_test["PassengerId"],
        "Survived": y_pred
    })
 submission.to_csv('../practice/submission.csv', index=False)






 