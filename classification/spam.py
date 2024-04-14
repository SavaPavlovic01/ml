from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from scipy.sparse import vstack
import os
import re
#from nltk.stem import PorterStemmer

num = True
url = True
to_lower = False
remove_header = True
first = True

def tokenizer(text):
    #ps = PorterStemmer()
    #tokens =  re.split("\\s+", text)
    #return ps.stem(tokens)
    if remove_header:
        text = re.split("\n\n", text)
        text = ''.join(text[1:])
    
    return re.split("\\s+", text)

def preproc_num(text):
    return re.sub("[-+]?[0-9]+", "NUMBER ", text)

def preproc_url(text):
    return re.sub("((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)", "URL ", text)

def preproc(text):
    if num:
        text = preproc_num(text)
    if url:
        text = preproc_url(text)
    return text

def get_tokenizer(punc = True):
    if punc:return tokenizer
    return None


if __name__ == "__main__":

    parent = os.path.join("data", "easy_ham")
    file_names = [os.path.join(parent, file) for file in os.listdir(parent)]

    vect = CountVectorizer(input='filename', decode_error='ignore', tokenizer=get_tokenizer(False), preprocessor=preproc, lowercase = to_lower)
    matrix = vect.fit_transform(file_names)

    print(pd.DataFrame(matrix.toarray(), columns=vect.get_feature_names_out()))