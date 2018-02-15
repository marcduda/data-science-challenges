#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:56:17 2018

@author: marcduda
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import re
import enchant

dictionary = enchant.Dict("fr_FR")
#import the output from ocr processing
directory = "statuts-ocr/"
files = os.listdir(directory)

#create the paragraphes and filter out the small ones (titles for example)
list_all_paragraphe = []
for file in files:
    if os.path.getsize(directory+file) > 0:
        ocr_txt = pd.read_csv(directory+file,header=None, delim_whitespace=True, quoting=csv.QUOTE_NONE, encoding='utf-8')
        ocr_txt.columns = ['page','x0','y0','x1','y1','word']
        ocr_txt['paragraphe'] = 0
        paragraphe = 0
        current_string = ""
        for i in range(1, len(ocr_txt)):
            word = re.sub(r'[^\w\s]',' ',str(ocr_txt.iloc[i, ocr_txt.columns.get_loc('word')]) )
            if abs(ocr_txt.iloc[i, ocr_txt.columns.get_loc('y0')]-ocr_txt.iloc[i-1, ocr_txt.columns.get_loc('y0')])>30: 
                paragraphe+=1
                list_all_paragraphe.append(current_string)
                if dictionary.check(word):
                    current_string =" "+word
                else:
                    current_string=" "
            elif dictionary.check(word):
                current_string+=" "+word
                
            ocr_txt.iloc[i, ocr_txt.columns.get_loc('paragraphe')]=paragraphe

list_all_paragraphe_filtered = [para for para in list_all_paragraphe if len(para)>20] 
 #%%Use NMF and LDA to find the number of topics and thus the number of cluster
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords # Import the stop word list
import nltk.data 
from gensim.utils import lemmatize
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from nltk.stem.snowball import SnowballStemmer

n_features = 1000
n_components = 50
n_top_words = 20

tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')


stemmer = SnowballStemmer("french")
full_stopwords = stopwords.words("french")+[ "ils", "elles", "les", "leurs","i","ii" ,"vi", "iv", "iii"]
list_all_paragraphe_filtered = [ ' '.join([stemmer.stem(y) for y in z.split(' ')]) for z in list_all_paragraphe_filtered]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=5,
                                   max_features=n_features,analyzer="word",
                                   stop_words = full_stopwords
                                   ,token_pattern='[^\d\W]{2,}')

tfidf = tfidf_vectorizer.fit_transform(list_all_paragraphe_filtered)
print("done vectorizer")
# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.80, min_df=5,
                                max_features=n_features,analyzer="word",
                                stop_words=full_stopwords
                                ,token_pattern='[^\d\W]{2,}')
tf = tf_vectorizer.fit_transform(list_all_paragraphe_filtered)
print("done with count vectorizer")

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (len(list_all_paragraphe_filtered), n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

print("processing dict and corpus")
list_all_paragraphe_split = [sentence.split() for sentence in list_all_paragraphe_filtered]
dictionary = Dictionary(list_all_paragraphe_split)
corpus = [dictionary.doc2bow(text) for text in list_all_paragraphe_split]

print("Topics in HDP model :")
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdptopics = hdpmodel.show_topics(formatted=False)
print("there are "+str(len(hdptopics))+" topics in the data")
alpha = hdpmodel.hdp_to_lda()[0]
plt.figure()
plt.plot(alpha)
plt.show()

# Fit the NMF model
print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
      "tf-idf features, n_samples=%d and n_features=%d..."
      % (len(list_all_paragraphe_filtered), n_features))
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done with nmf ")

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

#%%Embedd the paragraphs
#create word2vec model
print("starting paragraphe vectorization")
from gensim.models import word2vec, doc2vec
list_all_paragraphe_split = [sentence.split() for sentence in list_all_paragraphe_filtered]
labels = ["paragraph_"+str(i) for i in range(len(list_all_paragraphe_split))]
model = word2vec.Word2Vec(list_all_paragraphe_split, size=100)
print("done with word2vec")
#create doc2vec model
sentences = [doc2vec.LabeledSentence(words=sentence,tags=labels) for sentence in list_all_paragraphe_split ]#[str(list_all_paragraphe_split.index(sentence))]
model_doc = doc2vec.Doc2Vec(sentences,size=100)
print("done with doc2vec")
#store the model to  files

model_doc.save('my_model_complete_data.doc2vec')

#%%Do a dimension reduction on the data to see if it influences the clustering
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
sentences_vec = [model_doc.docvecs[label] for label in labels]
sentences_vec = normalize(sentences_vec)
pca = PCA(n_components="mle", svd_solver='full')
pca.fit(sentences_vec) 
print(pca.n_components_)
sentences_vec = pca.transform(sentences_vec) 
#plot inertia and silhouette for diffent cluster numbers
inertia = []
silhouette = []
silhouette_spec = []
for n_clusters in range(2,50,3):
    model = KMeans(n_clusters = n_clusters, random_state=10)
    model_spec = SpectralClustering(n_clusters = n_clusters, random_state=10)
    
    inertia.append(model.fit(sentences_vec).inertia_)
    
    cluster_labels = model.fit_predict(sentences_vec)
    silhouette_avg = silhouette_score(sentences_vec, cluster_labels)
    silhouette.append(silhouette_avg)
    cluster_labels_spec = model_spec.fit_predict(sentences_vec)
    silhouette_avg_spec = silhouette_score(sentences_vec, cluster_labels_spec)
    silhouette_spec.append(silhouette_avg_spec)
    
plt.figure()
plt.plot(range(2,50,3),inertia)
plt.show()

plt.figure()
plt.plot(range(2,50,3),silhouette)
plt.show()

plt.figure()
plt.plot(range(2,50,3),silhouette_spec)
plt.show()

#%%Deep learning using doc2vec embedding
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
nb_clusters = 10
model = KMeans(n_clusters = nb_clusters, random_state=10)

sentences_vec = [model_doc.docvecs[label] for label in labels]

sentences_vec = normalize(sentences_vec)
sentences_classif = np.asarray(sentences_vec)
cluster_labels = model.fit_predict(sentences_vec)

np.random.seed(10)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn import metrics

sentences_train,sentences_test,y_train,y_test = train_test_split(sentences_classif,cluster_labels, test_size=0.15, random_state=42)
dummy_y = np_utils.to_categorical(y_train)

dim_features = sentences_train.shape[1]
model_dl = Sequential()
model_dl.add(Dense(20, input_dim=dim_features, activation='relu'))
model_dl.add(Dense(10, activation='relu'))
model_dl.add(Dense(nb_clusters, activation='sigmoid'))

# Compile model
model_dl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['logloss'])

# Fit the model
model_dl.fit(np.asarray(sentences_train), dummy_y, epochs=20, batch_size=100)

# evaluate the model
scores = model_dl.evaluate(np.asarray(sentences_train), dummy_y)
print("\n%s: %.2f%%" % (model_dl.metrics_names[1], scores[1]*100))  

#predict the labels of the test set and print some metrics to compare it with the correct labels
predictions = model_dl.predict(np.asarray(sentences_test))
prediction_binary = [i for i in np.argmax(predictions,1)]
prediction_label = prediction_binary
print("DL part, metrics on test set:")
print(metrics.classification_report(y_test, prediction_label)) 

#%% Deep learning using word2vec embedding

from gensim.corpora.dictionary import Dictionary
def create_dictionaries(data = None,model = None):
    if (data is not None) and (model is not None) :
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(dataset):
            ''' Words become integers
            '''
            for key in dataset.keys():
                txt = dataset[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                dataset[key] = new_txt
            return data
        data = parse_dataset(data )
        return w2indx, w2vec, data
    else:
        print('No data provided...')
        
from keras.models import Sequential
from keras.layers import Dense,LSTM,SimpleRNN
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

top_words = 10000
max_paragraphe_length = 100
batch_size=128
embedding_vector_length = 100
y = cluster_labels
X = {i:list_all_paragraphe_filtered[i] for i in range(len(list_all_paragraphe_filtered))}

index_dict, word_vectors, X = create_dictionaries(data=X,model=model)
X = X.values()
X = sequence.pad_sequences(X, maxlen=max_paragraphe_length)
y = to_categorical(y)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.15, random_state=42)


def create_lstm(top_words,embedding_vector_length,max_review_length,batch_size):
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(LSTM(128, dropout=0.20, recurrent_dropout=0.20))
    model.add(Dense(10, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['logloss'])
    return model

def create_rnn(top_words,embedding_vector_length,max_review_length,batch_size):

    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(SimpleRNN( batch_input_shape=(1, 1, embedding_vector_length),units=1))
    model.add(Dense(10, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['logloss'])
    return model

def executeModel(str,X_train,y_train,X_test,y_test,top_words,embedding_vector_length,max_review_length,batch_size):
    if str=='rnn':
        model = create_rnn(top_words,embedding_vector_length,max_review_length,batch_size) 
    elif str=='lstm':
        model = create_lstm(top_words,embedding_vector_length,max_review_length,batch_size)
    model.fit(X_train, y_train, epochs=3, batch_size=batch_size, verbose=1,validation_split=0.1,shuffle=False)#,class_weight = class_weight_dl
    # Final evaluation of the model
    score,acc = model.evaluate(X_test, y_test, verbose=1)
    print("Score: %.2f%%\nLogloss: %.2f%%" % (score*100,acc*100))

    return model
    
modelRNN = executeModel('rnn',X_train,y_train,X_test,y_test,top_words,embedding_vector_length,max_paragraphe_length,batch_size)

modelLSTM = executeModel('lstm',X_train,y_train,X_test,y_test,top_words,embedding_vector_length,max_paragraphe_length,batch_size)
