import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, \
                            precision_recall_curve, plot_precision_recall_curve, \
                            average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib


from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer
from pandarallel import pandarallel
import re
import pickle
import joblib

import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool




def split_url(url):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    stemmer = SnowballStemmer(language="english")
    sent = tokenizer.tokenize(url)
    sent = [stemmer.stem(word) for word in sent]
    sent = ' '.join(sent)
    return sent

def vectorize(corpus, is_fitted=False, random_state=42):
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df = 3, 
                            lowercase=True, max_features=150000, 
                            stop_words=['url', 'label', 'stemmed', 'words_count', 'len'])
    if not is_fitted:
        url_vec = tfidf.fit_transform(corpus)
        joblib.dump(tfidf, 'models/tfidf.pkl') 
        svdT = TruncatedSVD(n_components=150, random_state=random_state)
        url_vec = svdT.fit_transform(url_vec)
        with open('models/svdt.pkl', 'wb') as fp:
            pickle.dump(svdT, fp) 
    else:
        print('loading existing models...')
        tfidf = joblib.load('models/tfidf.pkl')
        url_vec = tfidf.transform(corpus)
        with open('models/svdt.pkl', 'rb') as fp:
            svdT = pickle.load(fp)
        url_vec = svdT.transform(url_vec)
    return url_vec


def get_host_path(address):
    if not re.search(r'^[A-Za-z0-9+.\-]+://', address):
        address = 'tcp://{0}'.format(address)
    return urlparse(address)


def count_num(string):
    if string == None or string == '':
        return 0
    return len(re.sub("[^0-9]", "", string))


def add_features(df, is_fitted = False):
    df['stemmed'] = df['url'].parallel_apply(split_url)
    df['words_count'] = df['stemmed'].parallel_apply(lambda url: len(url.split()))
    #vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df = 3, lowercase=True, max_features=150000, stop_words=['url', 'label', 'stemmed', 'words_count', 'len'])
    #url_vec = vectorizer.fit_transform(df['stemmed'])
    #temp_df = pd.DataFrame(url_vec.toarray(), columns=vectorizer.get_feature_names())
    df['url_len'] = df['url'].parallel_apply(lambda url: len(url))    
    df['hostname'] = df['url'].parallel_apply(lambda url: get_host_path(url).hostname)
    df['path'] = df['url'].parallel_apply(lambda url: get_host_path(url).path)
    df[['hostname', 'path']].fillna(0, inplace=True)                                    
    df['hostname_len'] = df['hostname'].parallel_apply(lambda item: len(item) if item!=None else 0)  
    df['path_len'] = df['path'].parallel_apply(lambda item: len(item) if item!=None else 0)
    df['hostname_nums'] = df['hostname'].parallel_apply(count_num) 
    df['path_nums'] = df['path'].parallel_apply(count_num)                                
    df.drop(['hostname', 'path'], axis=1, inplace=True)
    
    url_vecs = pd.DataFrame(vectorize(df['stemmed'], is_fitted))
    df = pd.concat([df, url_vecs], axis=1)
    df.drop(['url', 'stemmed'], axis=1, inplace=True)
    return df
    
    
def split(df, random_state=42):
    train_text, temp_text, \
    train_labels, temp_labels = train_test_split(df,   
                                                 df['label'],
                                                 random_state = random_state,
                                                 test_size = 0.3,
                                                 stratify=df['label'])
    val_text, test_text, \
    val_labels, test_labels = train_test_split(temp_text,
                                              temp_labels,
                                              random_state = random_state,
                                              test_size = 0.5,
                                              stratify=temp_labels)
    
    return train_text.drop(['label'], axis=1), \
           train_labels, \
           val_text.drop(['label'], axis=1), \
           val_labels, \
           test_text.drop(['label'], axis=1), \
           test_labels


def preprocess(df, is_fitted = False):
    df = add_features(df, is_fitted)
    return split(df)


def train_gbm(train_data, train_labels, val_data, val_labels, test_data, test_labels, random_state=42):
    gbm = CatBoostClassifier(task_type="GPU", logging_level='Silent', 
                             loss_function='Logloss', od_type='Iter', od_wait=20, random_state=random_state)
    eval_pool = Pool(val_data, val_labels)
    gbm.fit(train_data, train_labels, eval_set=eval_pool, use_best_model=True)
    gbm.save_model('catboost_1',
                   format="cbm",
                   export_parameters=None,
                   pool=None)
    pred_probs = gbm.predict_proba(test_data)[:,1]
    pred_labels = gbm.predict(test_data)
    score = [roc_auc_score(test_labels, pred_probs), f1_score(test_labels, pred_labels)]
    
    print('roc_auc: ', score[0])
    print('f1: ', score[1])

    average_precision = average_precision_score(pred_labels, test_labels)

    disp = plot_precision_recall_curve(gbm, test_data, test_labels)
    disp.ax_.set_title('2-class Precision-Recall curve: ')
    
    return gbm

def get_cross_val(model, train_data, train_labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    cv_results = cross_val_score(model, train_data, train_labels, cv=skf, scoring='roc_auc')
    return cv_results