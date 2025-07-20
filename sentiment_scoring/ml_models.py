from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
import spacy
import multiprocessing as mp
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import TransformerMixin, BaseEstimator
import string
import re


import nltk
from nltk.stem import PorterStemmer

nlp = spacy.load('en_core_web_sm')
pd.set_option('display.max_colwidth',999)


class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=-1, Stemming=False):
        self.n_jobs = n_jobs
        self.Stemming = Stemming
        self.ps = PorterStemmer() if Stemming else None

    def fit(self, X, y):
        return self

    def transform(self, X):
        lower_case_text = X.apply(lambda x: x.lower())
        removed_punct_text = lower_case_text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        removed_numbers_text = removed_punct_text.apply(lambda x: re.sub(" \d+", " ", x))
        clear_whitespace_text = removed_numbers_text.apply(lambda x: re.sub(' +', ' ', x.lstrip().rstrip()))
        if self.Stemming and self.ps is not None:
            clear_whitespace_text = self.Stemdata(clear_whitespace_text)
        return clear_whitespace_text

    def Stemdata(self, X):
        text_tokens = X.apply(lambda x: x.split())
        stemed_tokens = text_tokens.apply(lambda x: [self.ps.stem(word) for word in x])
        return stemed_tokens.apply(lambda x: ' '.join(x))


class TextPreprocessor_withStem(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.ps = PorterStemmer()

    def fit(self, X, y):
        return self

    def transform(self, X):
        lower_case_text = X.apply(lambda x: x.lower())
        removed_punct_text = lower_case_text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        removed_numbers_text = removed_punct_text.apply(lambda x: re.sub(" \d+", " ", x))
        clear_whitespace_text = removed_numbers_text.apply(lambda x: re.sub(' +', ' ', x.lstrip().rstrip()))
        text_tokens = clear_whitespace_text.apply(lambda x: x.split())
        remove_stopwords_stem = text_tokens.apply(lambda x: [self.ps.stem(word) for word in x])

        return remove_stopwords_stem.apply(lambda x: ' '.join(x))


#importing financial phrase bank
train_data_path = r"./Financial sentiment analysis data/train_data.csv"
test_data_path = r"./Financial sentiment analysis data/test_data.csv"
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
print(train_data.shape)
print(test_data.shape)
X_train, y_train = train_data['sentence'], train_data['sentiment_label']
X_test, y_test   = test_data['sentence'],  test_data['sentiment_label']

# ------------------------------------------------------------------------------------------------------------------------
## Create LR model pipeline without stemming

preprocessor = TextPreprocessor()

vectorizer = CountVectorizer(analyzer='word',
                             token_pattern=r'\S+',
                             stop_words='english',
                             ngram_range=(1, 3),
                             binary=True)
func = f_classif
selector = SelectKBest(func, k=1000)

lr_classfier = LogisticRegression(random_state=42, max_iter=5000)

pipe_lr = Pipeline([('prep', preprocessor),
                    ('vec', vectorizer),
                    ('sel', selector),
                    ('clf', lr_classfier)
                    ])

# hyperparameter grid to search on
grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 0.5, 0.1],
                   'clf__solver': ['liblinear', 'saga'],

                   'sel__k': [1000, 5000, 10000, 20000, 'all'],

                   'sel__score_func': [f_classif, chi2],

                   'vec__ngram_range': [(1, 3), (2, 3), (1, 2)],
                   'vec__binary': [True, False],
                   'prep__Stemming': [True, False]}]

## Final LR model with Grid
LR_model = GridSearchCV(estimator=pipe_lr,
                        param_grid=grid_params_lr,
                        scoring='f1_macro',
                        cv=5,
                        n_jobs=-1)

### Performing gridsearch on LR model
LR_model.fit(X_train, y_train)

print('Best params are : %s' % LR_model.best_params_)
# Best training data accuracy
print('Best training f1_macro: %.3f' % LR_model.best_score_)

best_LR_model = LR_model.best_estimator_

#predict on test data
y_pred_lr = best_LR_model.predict(X_test)
print('Test Data classification report')
print(classification_report(y_test, y_pred_lr))

# -----------------------------------------------------------------------------------------------------------------------
# LR Model with Stemming
Stempreprocessor = TextPreprocessor_withStem()

pipe_lr_withStem = Pipeline([('prep', Stempreprocessor),
                             ('vec', vectorizer),
                             ('sel', selector),
                             ('clf', lr_classfier)
                             ])

## Final LR model with Grid
grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 0.5, 0.1],
                   'clf__solver': ['liblinear', 'saga'],

                   'sel__k': [1000, 5000, 10000, 20000, 'all'],

                   'sel__score_func': [f_classif, chi2],

                   'vec__ngram_range': [(1, 3), (2, 3), (1, 2)],
                   'vec__binary': [True, False]}]
LR_model_withStem = GridSearchCV(estimator=pipe_lr_withStem,
                                 param_grid=grid_params_lr,
                                 scoring='f1_macro',
                                 cv=5,
                                 n_jobs=-1)

### Performing gridsearch on LR model
LR_model_withStem.fit(X_train, y_train)

print('Best params are : %s' % LR_model_withStem.best_params_)
# Best training data accuracy
print('Best training f1_macro: %.3f' % LR_model_withStem.best_score_)

best_LR_model_withStem = LR_model_withStem.best_estimator_

#predict on test data
y_pred_lr_withStem = best_LR_model_withStem.predict(X_test)
print('Test Data classification report')
print(classification_report(y_test, y_pred_lr_withStem))


# -----------------------------------------------------------------------------------------------------------------------
# SVC Model without Stemming
svc_classifier = SVC(kernel='rbf',
                     C=100,
                     random_state=42)

pipe_svm = Pipeline([('prep', preprocessor),
                     ('vec', vectorizer),
                     ('sel', selector),
                     ('clf', svc_classifier)
                     ])

grid_params_svm = [{'clf__gamma': ['scale', 'auto'],
                    'clf__C': [100, 10, 1.0, 0.1, 0.01],
                    'clf__kernel': ['rbf', 'linear', 'poly'],

                    'sel__k': [1000, 5000, 10000, 20000, 'all'],

                    'sel__score_func': [f_classif, chi2],

                    'vec__ngram_range': [(1, 3), (2, 3), (1, 2)],
                    'vec__binary': [True, False]}]

SVC_model = GridSearchCV(estimator=pipe_svm,
                         param_grid=grid_params_svm,
                         scoring='f1_macro',
                         cv=5,
                         n_jobs=-1)

SVC_model.fit(X_train, y_train)

print('Best params are : %s' % SVC_model.best_params_)
# Best training data accuracy
print('Best training f1_macro: %.3f' % SVC_model.best_score_)

best_SVC_model = SVC_model.best_estimator_

#predict on test data
y_pred_svc = best_SVC_model.predict(X_test)
print('Test Data classification report')
print(classification_report(y_test, y_pred_svc))


# -----------------------------------------------------------------------------------------------------------------------
# Creating SVC model with Stemming
pipe_svm_withStem = Pipeline([('prep', Stempreprocessor),
                              ('vec', vectorizer),
                              ('sel', selector),
                              ('clf',svc_classifier)
                             ])

SVC_model_withStem = GridSearchCV(estimator  = pipe_svm_withStem,
                                  param_grid=grid_params_svm,
                                  scoring='f1_macro',
                                  cv=5,
                                  n_jobs=-1)

SVC_model_withStem.fit(X_train, y_train)

print('Best params are : %s' % SVC_model_withStem.best_params_)
# Best training data accuracy
print('Best training f1_macro: %.3f' % SVC_model_withStem.best_score_)

best_SVC_model_withStem = SVC_model_withStem.best_estimator_

#predict on test data
y_pred_svc_withStem = best_SVC_model_withStem.predict(X_test)
print('Test Data classification report')
print(classification_report(y_test, y_pred_svc_withStem))


### pickle and save all 4 models
import pickle
pickle.dump(best_LR_model, open('LR_model_withoutStem.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(best_LR_model_withStem, open('LR_model_withStem.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(best_SVC_model, open('SVC_model_withoutStem.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(SVC_model_withStem, open('SVC_model_withStem.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)