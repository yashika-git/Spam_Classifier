#!/usr/bin/env python
# coding: utf-8

# In[2]:



import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

lemmatizer = WordNetLemmatizer()


def text_preprocessing(text):
 # lowering all the words in the dataframe
 text = text.lower()

 # seperating all the words from the datafram
 text = nltk.word_tokenize(text)

 # removing all the stop words
 text = [i for i in text if i not in stopwords.words('english')]

 # removing all the words which are special characters and punctuations
 text = [i for i in text if i not in string.punctuation]
 text = [i for i in text if i.isalnum()]

 # stemming the text we get
 text = [lemmatizer.lemmatize(i) for i in text]
 return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
 #Preprocessing
 transformed_text = text_preprocessing(input_sms)

 #Vectorize
 vector_input = tfidf.transform([transformed_text])

 #Predict
 result = model.predict(vector_input)[0]

 if result==1:
     st.header("Spam")
 else:
     st.header("Not Spam")


# In[ ]:




