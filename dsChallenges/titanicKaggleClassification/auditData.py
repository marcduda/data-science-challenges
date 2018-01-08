#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:00:30 2017

@author: marcduda
"""
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

train_ = pd.read_csv('train.csv')
test_= pd.read_csv('test.csv')
test_['Survived'] = np.nan

data = pd.concat([train_,test_], axis=0)
#Validity
print(data['Sex'].unique())
print(data['Embarked'].unique())
print(data['SibSp'].unique())
print(data['Parch'].unique())
print(data['Pclass'].unique())
print(data['Cabin'].unique())
print(data['Fare'].unique())
print(len(data['PassengerId'].unique()))
print(data.loc[data['Fare']>500])
print(data['Age'].unique())
#print(data['Ticket'].unique())
data['TicketNumber'] = data['Ticket'].apply(lambda x : x.split(' ')[-1])
#print(data['TicketNumber'].unique())
print(len(data['Ticket'].unique()))
print(len(data['TicketNumber'].unique()))
#print(data.loc[data['Ticket']!=data['TicketNumber']])
def removeQuotes(string):
    if '"' in string:
        if len(string.split('"')[1]) >1:
            if '(' in string or ')' in string or len(string.split('"')[1].split(' '))>1:
                return string.replace('\"','')
            else:
                return ''.join([string.split('"')[0],string.split('"')[2]])
            
        else:
            return string.replace("\"","")
    else:
        return string
data['Name']=data['Name'].apply(lambda x: removeQuotes(x.replace('/','-')))
patternNameBuyer = re.compile("[\sa-z-']+,\s[a-z]+\.[\sa-z\.\/]+", re.IGNORECASE)
patternNameGuest = re.compile("[\sa-z-']+,[\sa-z]+\.[\sa-z\.]+\([a-z-'\.+\s]+\)", re.IGNORECASE)
data['Buyer']= data['Name'].apply(lambda x: int(patternNameBuyer.fullmatch(x)!=None))
data['Guest']= data['Name'].apply(lambda x:int(patternNameGuest.match(x)!=None))
#print(data['Name'].apply(lambda x:patternNameBuyer.match(x)).unique())

#Accuracy: not applicable we can't verify data info against other datasets

#Completeness

print(data.count())

print(data.loc[data['Embarked'].isnull()])
#people not related to any member on boat, have to guess embarked port with general data
#print(data.loc[data['Name'].str.contains("Martha Evelyn")])
#X = data[['Sex','Fare','Pclass','SibSp','Parch','Embarked']].loc[data['Embarked'].notnull()].dropna()
#Z = pd.concat([X.drop(['Sex','Embarked'],axis=1), pd.get_dummies(X['Sex'])], axis=1)
#RFclf = RandomForestClassifier(n_estimators=2000)
#RFclf.fit(Z,pd.factorize(X['Embarked'])[0])
#yyyy = pd.factorize(X['Embarked'])[0]
#X_p = data[['Sex','Fare','Pclass','Age','SibSp','Parch']].loc[data['Embarked'].isnull()].dropna()
#X_p = pd.concat([X_p.drop(['Sex'],axis=1), pd.get_dummies(X_p['Sex'])], axis=1)
#target_names = ['S','C','Q']
data["Embarked"] = data["Embarked"].astype('category')
data['Embarked']=data.groupby(['Sex','Pclass','SibSp'])['Embarked'].transform(lambda x: x.fillna(x.mode()[0]))
data.count()
#data[['Embarked']].loc[data['Embarked'].isnull()] = ['S','S']#target_names[RFclf.predict(X_p).reshape((2,1))]


#data['Embarked'] = (data['Embarked'].cat.codes.replace(-1, np.nan)
#                   .interpolate().astype(int).astype('category')
#                   .cat.rename_categories(data['Embarked'].cat.categories))

#print(data.loc[data['Fare'].isnull()])
#print(data.loc[data['TicketNumber']=='3701'])
#print(data.shape)
data["Fare"] = data.groupby(['Sex','Pclass','Embarked','SibSp'])['Fare'].transform(lambda x: x.fillna(x.mean()))
data['Age'] = data.groupby(['Sex','Pclass','Embarked'])['Age'].transform(lambda x: x.fillna(x.mean()))
#print(data.loc[data['Fare'].isnull()])

#data["Age"] = data["Age"].astype('float')
#print(data.loc[(data['Age']<18) & (data['Sex']=='male')])
print(data.loc[data['Name'].str.contains('Dr\.')])
print(data['Name'].apply(lambda x: x.split(",")[1].split(".")[0]).unique())
def getTitle(string):
    title = string.split(",")[1].split(".")[0]
    if title in ['Don','Lady','Sir','the Countess','Jonkheer','Dona']:
        return 'Noble'
    elif title in ['Major','Col','Capt']:
        return 'Military'
    elif title in ['Miss','Mlle','Ms']:
        return 'Miss'
    elif title in ['Mme','Mrs']:
        return 'Mrs'
    else:
        return title
data['Title'] = data['Name'].apply(lambda x: getTitle(x))
#data['FamillySize'] = 
#print(data['Name'].loc[data['SibSp']+data['Parch']==0].unique())
def verifySex(df):
    if (df['Name'].split(",")[1].split(".")[0].strip() in ['Don','Sir','Jonkheer','Rev','Major','Col','Capt','Mr','Master']) and (df['Sex']=='male'):
        return True
    elif (df['Name'].split(",")[1].split(".")[0].strip() in ['Dona','Lady','the Countess','Mrs','Miss','Mme','Ms','Mlle']) and (df['Sex']=='female'):
        return True
    else:
        return False
data['ConsistentSex'] = data.apply(lambda df: int(verifySex(df)),axis=1)
data['Surname'] = data['Name'].apply(lambda x: x.split(",")[0].strip())
print(data['TicketNumber'].value_counts())
countTicket = data['TicketNumber'].value_counts()
def changedName(df):
    if  ('(' in df['Name']):
        return True
    else:
        return False
data['ChangedName']= data.apply(lambda x: int(changedName(x)),axis=1)
print(data[['Pclass','Name','ChangedName','Title']].loc[(data['ChangedName']==1) & (data['Title']=='Miss')])
def isMaidOrWorker(df):
    if (df['DifferenceReservation']>0) & (df['SibSp']==0) & (df['Parch']==0):
        return True
    else:
        return False
data['NbReservations']=data['TicketNumber'].apply(lambda x:countTicket[x])
data['NbMembers']=data['SibSp']+data['Parch']+1.
data['DifferenceReservation']= data['NbReservations']-data['NbMembers']
data['MaidOrWorker']=data.apply(lambda df: int(isMaidOrWorker(df)),axis=1)
#print(data[['Pclass','Name','SibSp','Parch','Maid']].loc[data['Maid']==1])
#false name or maiden name in brakes
#see if Mrs or Miss instead of Mrs
print(data[['Name','SibSp','Parch','MaidOrWorker','TicketNumber']].loc[data['MaidOrWorker']==1].sort_values('TicketNumber'))
def getDeck(x):
    if str(x)[0] is 'n':
        return None
    else:
        return str(x)[0]
data['Cabin'] = data['Cabin'].astype(str).replace(pd.np.nan,None)
data['Deck']=data['Cabin'].apply(lambda x: getDeck(x))
data['Deck']=data['Deck'].replace('n',None)
data['FareByPassenger']=data['Fare']/data['NbReservations']
print(data[['Deck','Cabin']].loc[data['Deck'].notnull()])
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
lb = preprocessing.LabelBinarizer()
data['SexBinary']= lb.fit_transform(data['Sex'])
dummies_Embarked = pd.get_dummies(data['Embarked'])
data_feature = data[['Age','FareByPassenger','Parch','Pclass','SibSp','MaidOrWorker','Survived','SexBinary']]#'Buyer','Guest','ChangedName',
data_feature = pd.concat([data_feature,dummies_Embarked],axis=1)

X = data_feature.loc[data_feature['Survived'].notnull()].drop(['Survived'],axis=1)
y = data_feature['Survived'].loc[data_feature['Survived'].notnull()]
data_feature_predict = data[['Age','FareByPassenger','Parch','Pclass','SibSp','MaidOrWorker','Survived','SexBinary','PassengerId']]#'Buyer','Guest','ChangedName',
data_feature_predict = pd.concat([data_feature_predict,dummies_Embarked],axis=1)
X_predict = data_feature_predict.loc[data_feature_predict['Survived'].isnull()].drop(['Survived'],axis=1)

X_, X_test, y_, y_test = train_test_split(X, y, test_size=.3, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, random_state=0)

#RFclf = RandomForestClassifier(warm_start=True, oob_score=True,max_features="auto",random_state=12)

forest_params = {'n_estimators':range(10,201,10),}#'max_features':['auto','sqrt','log2'] #'oob_score': [True, False],'class_weight':[{0: 1, 1: 10},{0: 1, 1: 5}]
rfc = GridSearchCV(RandomForestRegressor(max_features='auto'), forest_params, cv=5)
rfc.fit(X_train, y_train)

print(rfc.best_score_)                                 

for param_name in sorted(forest_params.keys()):
    print("%s: %r" % (param_name, rfc.best_params_[param_name]))


rf_scores = cross_val_predict(rfc.best_estimator_, X_train, y_train, cv=5)
print(rf_scores)
print(rf_scores.mean())


predicted_RF = rfc.predict(X_test)
print("RF part, metrics on test set:")
print(metrics.classification_report(y_test, pd.Series(predicted_RF>0.5).apply(lambda x: int(x))))

X_predict['Survived'] = rfc.predict(X_predict.drop(['PassengerId'],axis=1))
X_predict[['PassengerId','Survived']].to_csv('prediction.csv',index=False)


