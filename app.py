#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle


# In[ ]:


from flask import Flask,render_template,url_for,request
import pickle
import joblib
#filename = "pickle.pkl"
clf = pickle.load(open('mb_model.pkl','rb'))
cv=pickle.load(open('tfidf.pkl','rb'))
#tf=TfidfVectorizer(encoding="latin-1",strip_accents="unicode")


# In[ ]:


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data)
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run()


# 
