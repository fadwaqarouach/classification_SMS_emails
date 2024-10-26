# Import libraries
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


class AddFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
      nltk.download('punkt', quiet=True)
    
    def count_words(self, input_text):
        # remove punctuation, tokenize and return the number of tokens (words)
        message = input_text.translate(str.maketrans('', '', string.punctuation))
        return len(nltk.word_tokenize(input_text))

    def count_sentences(self, input_text):
        return len(nltk.sent_tokenize(input_text.lower()))

    def count_brackets(self, input_text):
        return len(re.findall(r'<[a-zA-Z0-9\s]+>+', input_text.lower()))

    def count_links(self, input_text):
        return len(re.findall(r'https?://\S+|www\.\S+', input_text.lower()))

    def count_phone(self, input_text):
        return len(re.findall(r'\d{5,}', input_text.lower()))

    def count_money(self, input_text):
        return len(re.findall(r'[$|£|€]\d+', input_text.lower()))+len(re.findall(r'\d+[$|£|€]', input_text.lower()))

    def transform(self, df, y=None):
        df['word_count'] = df.message.apply(self.count_words)
        df['sentence_count'] = df.message.apply(self.count_sentences)
        df['brackets_count'] = df.message.apply(self.count_brackets)
        df['links_count'] = df.message.apply(self.count_links)
        df['phone_count'] = df.message.apply(self.count_phone)
        df['money_count'] = df.message.apply(self.count_money)
        return df
    def fit(self, df, y=None):
        return self




# Class derived from BaseEstimator and TransformerMixin (sklearn) classes.
class TextPreprocess (BaseEstimator, TransformerMixin):
    def __init__(self):
        nltk.download('stopwords', quiet=True) # Download stopwords

    def to_lower(self, text):
        return str(text).lower()
    
    def replace_brackets(self, input_text):
        # Replace text between brackets with 'bracketstext' (spam messages)
        return re.sub('<.*?>+', ' brackets_text ', input_text)

    def replace_money(self, input_text):
        # Replace money amounts ($123 or 1£) with 'moneytext'
        input_text = re.sub(r'[$|£|€]\d+', ' money_text ', input_text)
        return re.sub(r'\d+[$|£|€]', ' money_text ', input_text)

    def replace_currency(self, input_text):
        # Replace remaining currency symbols with 'currsymb'
        return re.sub(r'[$|£|€]', ' curr_symb ', input_text)

    def replace_urls(self, input_text):
        # Replace links with 'weblink'
        link_regex = r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
        link_regex1 = r'https?://\S+|www\.\S+'
        link_regex2 = r'http.?://[^\s]+[\s]?'
        return re.sub(link_regex1, ' weblink ', input_text)

    def remove_punc(self, text):
        nopunc = [char for char in text if char not in string.punctuation]
        return "".join(nopunc)

    def remove_stopwords(self, text):
        nostop =   [
                    word
                    for word in text.split()
                    if word.lower() not in stopwords.words("english") and word.isalpha()
                    ]
        return nostop

    def transform(self, df, **transform_params):
        clean_txt = df.apply(self.to_lower)
        clean_txt = clean_txt.apply(self.replace_brackets)
        clean_txt = clean_txt.apply(self.replace_money)
        clean_txt = clean_txt.apply(self.replace_currency)
        clean_txt = clean_txt.apply(self.replace_urls)
        clean_txt = clean_txt.apply(self.remove_punc)
        clean_txt = clean_txt.apply(self.remove_stopwords)
        clean_txt = clean_txt.agg(lambda x: " ".join(map(str, x)))
        return clean_txt

    def fit(self, X, y=None, **fit_params):
        return self




class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):        
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self
    
textcountscols = ['word_count', 'sentence_count', 'brackets_count', 'links_count', 'phone_count', 'money_count']
