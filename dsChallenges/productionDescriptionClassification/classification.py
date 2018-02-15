#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:15:14 2017

@author: marcduda
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords 
import nltk.data 
from sklearn.utils import shuffle
import nltk.stem

train = pd.read_csv('train_data.csv',sep=';',
                    dtype={'Identifiant_Produit': int, 'prix': str
                            ,'Produit_Cdiscount': str,'Marque':str
                            ,'Libelle':str,'Description':str,'Categorie1':str
                            ,'Categorie2':str,'Categorie3':str})
        
test = pd.read_csv('test_data_final.csv',sep=';',
                   dtype={'Identifiant_Produit': int, 'prix': str
                           ,'Produit_Cdiscount': str,'Marque':str
                           ,'Libelle':str,'Description':str})

test['Categorie1'] = np.nan
test['Categorie2'] = np.nan
test['Categorie3'] = np.nan

data = pd.concat([train,test], axis=0)
#%%
data = data.loc[data['Produit_Cdiscount']!='Produit_Cdiscount']
print(data.loc[data['prix']==data['prix'].max()])
negativePrice = data.loc[data['prix'] =='-1.0']
NullPrice = data.loc[data['prix'] =='0.0']
tooBigPrice1 = data.loc[data['prix'] =='999999.99']
tooBigPrice2 = data.loc[data['prix'] =='99999.99']
tooBigPrice3 = data.loc[data['prix'] =='9999.99']
print(negativePrice.shape)
print(NullPrice.shape)
print(tooBigPrice1.shape)
print(tooBigPrice2.shape)
print(tooBigPrice3.shape)

#%%
print(data.dtypes)
marquesNull = data['Marque'].loc[data['Marque']!=data['Marque']].shape
libellesNull = data['Libelle'].loc[data['Libelle']!=data['Libelle']].shape
descriptionsNull = data['Description'].loc[data['Description']!=data['Description']].shape
cdiscountNull = data['Produit_Cdiscount'].loc[data['Produit_Cdiscount']!=data['Produit_Cdiscount']].shape
libelles = data['Libelle'].unique()


#%%
tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    



french_stemmer = nltk.stem.SnowballStemmer('french')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([french_stemmer.stem(w) for w in analyzer(doc)])

vectorizer_s = StemmedCountVectorizer(ngram_range=(1, 2),min_df=3
                                     ,analyzer="word"
                                     , stop_words= stopwords.words("french")
                                     ,tokenizer=LemmaTokenizer())
    
#apply 
data['Marque'].fillna(value='',inplace=True)
data['Text']=data['Marque']+" "+data['Description']

X = data[['Text','Categorie3']].loc[data['Categorie3'].notnull()&data['Text'].notnull()].drop(['Categorie3'],axis=1).values.reshape((-1,))
y = data['Categorie3'].loc[data['Categorie3'].notnull()&data['Text'].notnull()].values.reshape((-1,))

X, y = shuffle(X,y, random_state=0)

X_, X_test, y_, y_test = train_test_split(X[:100000], y[:100000], test_size=.1, random_state=0)#


X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, test_size=.2, random_state=0)
#%%

text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),min_df=5, stop_words= stopwords.words("french") )), #vectorizer_s ),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=10, tol=None))])

text_clf.fit(X_train, y_train)

predicted_SVM = text_clf.predict(X_test)
print("SVM part, metrics on test set:")
print(metrics.classification_report(y_test, predicted_SVM))

from sklearn.model_selection import GridSearchCV
parameters = {'clf__max_iter':(5,10,15),
              }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_) 

#%%
X_to_predict = data[['Text','Categorie3']].loc[data['Categorie3'].isnull()].drop(['Categorie3'],axis=1).values.reshape((-1,))
prediction_SVM = text_clf.predict(X_to_predict)
print("SVM part, metrics on test set:")
test_label_df = pd.DataFrame(data=prediction_SVM,columns=['label'])
test_id_df = data[['Identifiant_Produit','Categorie3']].loc[data['Categorie3'].isnull()].drop(['Categorie3'],axis=1)
data_test = pd.concat([test_id_df,test_label_df],axis=1)
data_test.to_csv('prediction_description.csv',index=False)