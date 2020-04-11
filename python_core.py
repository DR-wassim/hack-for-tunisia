#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier


# In[2]:


new_data = pd.read_csv("truthfy.csv")
new_data.info()
new_data.head()


# In[3]:


for i in range(0,new_data.shape[0]-1):
    if(new_data.Body.isnull()[i]):
        new_data.Body[i] = new_data.Headline[i]
        
y = new_data['Label']
X = new_data['Body']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[4]:


#Applying tfidf to the data set
tfidf_vect = TfidfVectorizer(stop_words = 'english')
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vect.get_feature_names())


# In[9]:


#Applying Naive Bayes
clf = MultinomialNB() 
clf.fit(tfidf_train, y_train)                      
pred = clf.predict(tfidf_test)                     
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)


# In[16]:


#Applying Passive Aggressive classifier
linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)


# In[11]:


new_data['Label'].value_counts()


# In[ ]:




