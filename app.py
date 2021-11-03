#!/usr/bin/env python
# coding: utf-8

# In[2]:




import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i not in stopwords.words('english')]
    text = [i for i in text if i not in string.punctuation]
    text = [i for i in text if i.isalnum()]
    text = [lemmatizer.lemmatize(i) for i in text]
    return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_text = text_preprocessing(input_sms)
    vector_input = tfidf.transform([transformed_text])
    result = model.predict(vector_input)[0]

    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# In[ ]:




