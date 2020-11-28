# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 09:39:40 2020

@author: lakhan
"""
#importing libraries
import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt

#importing the data
airline = pd.read_csv('airline_sentiment_analysis.csv')

airline.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow"])
   
    
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', airline['airline_sentiment'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = airline.iloc[:, -1].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
corpus = np.array(ct.fit_transform(X))

#training the datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus, X, test_size=0.2, random_state=0)

#classifier
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators = 200, oob_score = 'TRUE', n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)

#predictions
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
matrix = confusion_matrix( y_test.argmax(axis=1), predictions.argmax(axis=1))
sentiment = classification_report(y_test ,predictions)
accuracy = accuracy_score(y_test, predictions)
print(matrix)
print(sentiment)
print(accuracy)
