import pandas as pd
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from utils import *

ps = PorterStemmer()

def predict(model, data):
    df = pd.DataFrame(data, columns=['message'])
    af = AddFeatures()
    tp = TextPreprocess()

    df_counts = af.transform(df)
    df_clean = tp.transform(df.message)
    df = df_counts
    df['clean_msg'] = df_clean
    
    predictions = model.predict(df).tolist()

    return predictions


model = pickle.load(open('/app/app/model.pkl','rb'))
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):

    # 1. predict
    result = predict(model, [input_sms])[0]
    # 3. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
