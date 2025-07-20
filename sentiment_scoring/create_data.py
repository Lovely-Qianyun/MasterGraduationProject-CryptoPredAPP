from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
import string
import spacy
import re
import multiprocessing as mp
from sklearn.metrics import classification_report

import nltk
from nltk.stem import PorterStemmer

nlp = spacy.load('en_core_web_sm')
pd.set_option('display.max_colwidth',999)
## Load all paths

#importing financial phrase bank
financial_phrasebank_file_name = r"./Financial sentiment analysis data/all-data.csv"

## different files from Sem Eval Dataset
semeval2_2017_train_file_name  = r"./Financial sentiment analysis data/Headline_Trainingdata.json"
semeval2_2017_test_file_name   = r"./Financial sentiment analysis data/Headlines_Testdata.json"
semeval2_2017_train_microblog_file_name   = r"./Financial sentiment analysis data/Microblog_Trainingdata.json"
semeval2_2017_test_microblog_file_name   = r"./Financial sentiment analysis data/Microblogs_Testdata.json"

# semeval2_2017_trial_file_name  = os.path.join('data','Project','Headline_Trialdata.json')
import chardet

with open(financial_phrasebank_file_name, 'rb') as f:
    result = chardet.detect(f.read())

df1 = pd.read_csv(financial_phrasebank_file_name, header=None,
                 names=['label','sentence'], encoding=result['encoding'])
print(result['encoding'])


df1.rename(columns={'label':'sentiment_label'}, inplace=True)
print('Shape of financial phrase bank dataset ', df1.shape[0])
df1['source'] = 'financialphrasebank'
print(df1.head())

df2_headline_train = pd.read_json(semeval2_2017_train_file_name)
df2_headline_test = pd.read_json(semeval2_2017_test_file_name)
df2_headline = pd.concat([df2_headline_train, df2_headline_test]).reset_index()
df2_headline.rename(columns={'title':'sentence', 'sentiment':'sentiment_score'}, inplace=True)
df2_headline = df2_headline[['sentence','sentiment_score']]
df2_headline['source'] = 'headline'
df2_headline['sentiment_label'] = df2_headline['sentiment_score'].apply(lambda x: 'positive' if x>0 else ('negative' if x<0 else 'neutral' ))
print('Shape of SemEval 2017 Headline bank dataset ', df2_headline.shape[0])
print(df2_headline.head())

print(df2_headline.sentiment_label.value_counts())

final_df = pd.concat([df1, df2_headline.drop('sentiment_score', axis=1)])
print(final_df.head())

print('Data label counts in the final dataset')
print(final_df.sentiment_label.value_counts())

from sklearn.model_selection import train_test_split
train_data, test_data= train_test_split(final_df,  test_size=0.2, random_state=42)

train_data.to_csv("./Financial sentiment analysis data/train_data.csv", index=False)
test_data.to_csv("./Financial sentiment analysis data/test_data.csv", index=False)

# 8:1:1
train_set, temp_test_set = train_test_split(final_df, test_size=0.2, random_state=42)
val_set, test_set = train_test_split(temp_test_set, test_size=0.5, random_state=42)

train_set.to_csv("./Financial sentiment analysis data/train_set.csv", index=False)
val_set.to_csv("./Financial sentiment analysis data/validation_set.csv", index=False)
test_set.to_csv("./Financial sentiment analysis data/test_set.csv", index=False)