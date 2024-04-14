import email.parser
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from scipy.sparse import vstack
import os
import re
import tarfile
import urllib.request
import email
import email.policy
from html import unescape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score
#from nltk.stem import PorterStemmer

num = True
url = True
to_lower = False
remove_header = True
first = True

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("data", "spam")

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()

def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

def fetch_spam_data(ham_url = HAM_URL, spam_url = SPAM_URL, spam_path = SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

def tokenizer(text):
    #ps = PorterStemmer()
    #tokens =  re.split("\\s+", text)
    #return ps.stem(tokens)
    parsed = email.parser.Parser(policy=email.policy.default).parsestr(text)
    text = email_to_text(parsed)
    if text == None: return ""
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

    ham_path = os.path.join("data", "spam", "easy_ham")
    spam_path = os.path.join("data", "spam", "spam")

    ham_files = [os.path.join(ham_path, file) for file in os.listdir(ham_path)]
    spam_files = [os.path.join(spam_path, file) for file in os.listdir(spam_path)]

    X = np.array(ham_files + spam_files, dtype=object)
    Y = np.array([0] * len(ham_files) + [1] * len(spam_files))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    vect = CountVectorizer(input='filename', decode_error='ignore', tokenizer=get_tokenizer(True), preprocessor=preproc, lowercase = to_lower)
    train_vector = vect.fit_transform(X_train)
    test_vector = vect.transform(X_test)
    #vect_test = CountVectorizer(input='filename', decode_error='ignore', tokenizer=get_tokenizer(True), preprocessor=preproc, lowercase = to_lower)
    #test_vector = vect_test.fit_transform(X_test)
    #print(pd.DataFrame(train_vector.toarray(), columns=vect.get_feature_names_out())["This"])
    log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    

    log_clf.fit(train_vector, y_train)

    y_pred = log_clf.predict(test_vector)

    print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
    print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

# CV results
# [CV] END ................................ score: (test=0.983) total time=   0.7s
# [CV] END ................................ score: (test=0.976) total time=   0.3s
# [CV] END ................................ score: (test=0.986) total time=   1.0s

# On test set
# Precision: 100.00%
# Recall: 94.23%