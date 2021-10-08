#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 00:16:08 2021

@author: pengxy
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("/Users/pengxy/Downloads/hw4/data_train_hw4_problem1.csv")
data.head()

# Checking how many of them are null
data.isnull().sum()

# convert label to a numerical variable
data['numspam'] = data['spam'].map({False:0, True:1})

# Count the number of words in each Text
data['Count']=0
for i in np.arange(0,len(data.text)):
    data.loc[i,'Count'] = len(data.loc[i,'text'])

# Unique values in target set
print("Unique values in the spam set: ", data.spam.unique())

# displaying the new table
data.head()

# collecting non-spam messages in one place 
nonspam  = data[data.numspam == 0]
nonspam_count  = pd.DataFrame(pd.value_counts(nonspam['Count'],sort=True).sort_index())
print("Number of nonspam messages in data set:", nonspam['spam'].count())
print("nonspam Count value", nonspam_count['Count'].count())

# collecting spam messages in one place 
spam = data[data.numspam == 1]
spam_count = pd.DataFrame(pd.value_counts(spam['Count'],sort=True).sort_index())
print("Number of spam messages in data set:", spam['spam'].count())
print("Spam Count value:", spam_count['Count'].count())

fig, ax = plt.subplots(figsize=(17,5))
spam_count['Count'].value_counts().sort_index().plot(ax=ax, kind='bar',facecolor='red');
nonspam_count['Count'].value_counts().sort_index().plot(ax=ax, kind='bar',facecolor='green');

import nltk
import os

#if true it will download all the stopwords
if True:
    os.system('python -m nltk.downloader')
    
# importing Natural Language Toolkit 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#if true will create vectorizer with stopwords
if True:
    stopset = set(stopwords.words("english"))
    vectorizer = TfidfVectorizer(stop_words=stopset,binary=True)

#if true will create vectorizer without any stopwords
if True:
    vectorizer = TfidfVectorizer()
# Extract feature column 'Text'
X = vectorizer.fit_transform(data.text)
# Extract target column 'Class'
y = data.numspam
#Shuffle and split the dataset into the number of training and testing points
if True: 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# Import the models from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.model_selection import KFold

objects = ('Multi-NB', 'DTs', 'AdaBoost', 'KNN', 'RF')

# function to train classifier
def train_classifier(clf, X_train, y_train):    
    clf.fit(X_train, y_train)

# function to predict features 
def predict_labels(clf, features):
    return(clf.predict(features))

# Initialize the three models
A = MultinomialNB(alpha=1.0,fit_prior=True)
B = DecisionTreeClassifier(random_state=42)
C = AdaBoostClassifier(n_estimators=100) 
D = KNeighborsClassifier(n_neighbors=1)
E = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
# loop to call function for each model
clf = [A,B,C,D,E]
pred_val = [0,0,0,0,0]

for a in range(0,5):
    train_classifier(clf[a], X_train, y_train)
    y_pred = predict_labels(clf[a],X_test)
    pred_val[a] = f1_score(y_test, y_pred) 
    print(pred_val[a])
    
# ploating data for F1 Score
y_pos = np.arange(len(objects))
y_val = [ x for x in pred_val]
plt.bar(y_pos,y_val, align='center', alpha=0.7)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy Score')
plt.title('Accuracy of Models')
plt.show()

# defining the variable for learning curve
size, score, cv = np.linspace(.1, 1.0, 5), 'f1', KFold(n_splits= 5, shuffle = True,)

# calling the learning_curve function from defined variables
size, train, test = learning_curve(C, X, y, cv= cv, scoring=score, n_jobs=1, train_sizes=size)

# Mean and standard deviation of train and test score
train_mean,test_mean  =  np.mean( train, axis=1), np.mean( test, axis=1)
train_std,  test_std  =  np.std(train, axis=1) , np.std(test, axis=1)

# Ploating the Grid
plt.grid()

# Ploating the curve 
plt.fill_between(size, train_mean - train_std, train_mean + train_std, alpha=0.1,color="r")
plt.fill_between(size,  test_mean - test_std,   test_mean + test_std,  alpha=0.1,color="g")

# Ploating the axis name and legend 
plt.plot(size, train_mean, 'o-', color="r",label="Training score")
plt.plot(size, test_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")