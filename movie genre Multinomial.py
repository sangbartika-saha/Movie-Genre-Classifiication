#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install SVM')


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk 
from nltk.corpus import stopwords #for cleaning 
from nltk.stem import LancasterStemmer ##for cleaning 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 


# In[2]:


train_path="C:/Users/Sangbartika Saha/Desktop/projects/Internship/Genre/Genre Classification Dataset/train_data.txt"
train_data = pd.read_csv(train_path, sep=":::" , names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")


# In[3]:


train_data


# In[4]:


test_path="C:/Users/Sangbartika Saha/Desktop/projects/Internship/Genre/Genre Classification Dataset/test_data.txt"
test_data = pd.read_csv(test_path, sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")


# In[5]:


test_data


# In[6]:


train_data.head(250)


# In[7]:


train_data.info()


# In[8]:


test_data.info()


# In[9]:


train_data.describe()


# In[10]:


train_data.isnull()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[12]:


X_train,X_test,y_train,y_test = train_test_split(train_data['TITLE'],train_data['GENRE'],test_size=0.2,random_state=42)


# In[13]:


vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[14]:


clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)


# In[15]:


y_pred = clf.predict(X_test_tfidf)


# In[16]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[17]:


from sklearn.metrics import accuracy_score, classification_report


# In[18]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[23]:


new_titles = ["The Secret Sin ", "A heartwarming love story set in Paris."]
new_titles_tfidf = vectorizer.transform(new_titles)
predicted_genres = clf.predict(new_titles_tfidf)


# In[24]:


print("\nPredicted Genres for New Synopses:")
for titles, genre in zip(new_titles, predicted_genres):
    print(f"Synopsis: {titles}\nPredicted Genre: {genre}\n")


# In[25]:


new_synopses = ["Cupid"]
new_synopses_tfidf = vectorizer.transform(new_synopses)
predicted_genres = clf.predict(new_synopses_tfidf)


# In[26]:


print("\nPredicted Genres for New Synopses:")
for titles, genre in zip(new_synopses, predicted_genres):
    print(f"Synopsis: {titles}\nPredicted Genre: {genre}\n")


# In[ ]:




