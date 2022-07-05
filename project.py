import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # text conversion to lower case
    text = nltk.word_tokenize(text)  # tokenization
    # removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    # stopwords & punctuations
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    # stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('VectorizerN.pkl','rb'))
model = pickle.load(open('ModelN.pkl','rb'))

st.title("Email Spam Classifier")
input_msg = st.text_input("Enter the text")
if st.button('Predict'):
    # 1. pre-process
    transformed_text = transform_text(input_msg)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")





