import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import warnings
warnings.filterwarnings("ignore")
import urllib.request
import pickle
import gc
from transformer import TransformerEncoderLayer, TransformerClassifier, labelencoder, macro_f1
from transformers import TFBertModel, BertTokenizer
from sklearn.base import TransformerMixin, BaseEstimator
from nltk.stem import PorterStemmer
import string
import re
import time
import random
import ssl

import threading


def fetch_url_with_timeout(url, headers, timeout=30):
    result = {'text': None, 'error': None}

    def _fetch():
        try:
            req = urllib.request.Request(url, headers=headers)
            context = ssl._create_unverified_context()
            with urlopen(req, context=context, timeout=10) as response:
                html = response.read()  # 读取响应内容
            soup = BeautifulSoup(html, features="html.parser")  # 解析HTML

            for script in soup(["script", "style"]):  # kill all script and style elements
                script.decompose()
            text = soup.get_text()  # get text
            lines = (line.strip() for line in
                     text.splitlines())  # break into lines and remove leading and trailing space on each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  # break multi-headlines into a line each
            result['text'] = '\n'.join(chunk for chunk in chunks if chunk)
        except Exception as e:
            result['error'] = str(e)

    thread = threading.Thread(target=_fetch)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        result['error'] = f"Request timed out after {timeout} seconds"
        return result
    return result

def create_dataset(df, tokenizer, max_len, batch_size):
    sentences = df['sentence'].values

    inputs = tokenizer(
        sentences.tolist(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    return tf.data.Dataset.from_tensor_slices((inputs, sentences)).batch(batch_size)

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

# Loading Previous Package + Reading in the Model Fitted
from joblib import load
svc_model = load(r'./SVC_model_withStem.pkl', 'rb')
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f'Using {device}')
bert_model_weight = 'ProsusAI/finbert'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_weight)    #tokenizer to use
bert_model = TFBertModel.from_pretrained(bert_model_weight)    #loading base bert model


MAX_LEN = 32
NUM_LAYERS = 4
D_MODEL = 768
NUM_HEADS = 12
DFF = 3072
DROPOUT_RATE = 0.1

transformer_model = TransformerClassifier(
    bert_model=bert_model,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    max_len=MAX_LEN,
    rate=DROPOUT_RATE
)

optimizer = Adam(learning_rate=5e-5)
loss_fn = SparseCategoricalCrossentropy()
transformer_model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[SparseCategoricalAccuracy()]
)

# test
test_input = bert_tokenizer(
    "test",
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)

_ = transformer_model(test_input, training=False)
transformer_model.load_weights('./FinBERT-Transformer.h5')

BATCH_SIZE = 32
MAX_LEN = 32

# Creating Table for Sentiment Score
start_date = pd.to_datetime('2013-04-01')
end_date = pd.to_datetime('2021-09-30')
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
year_day = [date.year for date in date_range]
month_day = [date.month for date in date_range]
day_day = [date.day for date in date_range]

# column_name = ['positive', 'neutral', 'negative', 'exception']
column_name = ['positive', 'neutral', 'negative']
sentiment_score_svc = pd.DataFrame(
    np.zeros((len(date_range), len(column_name))),
    index=[year_day, month_day, day_day],
    columns=column_name
).reset_index().rename(
    columns={"level_0": "year", "level_1": "month", "level_2": "day"}
).set_index(['year', 'month', 'day'])

sentiment_score_trans = pd.DataFrame(
    np.zeros((len(date_range), len(column_name))),
    index=[year_day, month_day, day_day],
    columns=column_name
).reset_index().rename(
    columns={"level_0": "year", "level_1": "month", "level_2": "day"}
).set_index(['year', 'month', 'day'])

# Reading the List of files + Reading in the Text + Sentiment Scoring
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from bs4 import BeautifulSoup
import sys, time

READ_FREQ = 30
SAVING_FREQ = 10

def predict_unseen_test(model, unseen_test_dataset):
    model.trainable = False

    sentences = []
    predictions = []

    for batch in unseen_test_dataset:
        batch_inputs = batch[0]
        batch_sentences = batch[1]


        batch_preds = model(batch_inputs, training=False)
        batch_preds = tf.argmax(batch_preds, axis=1)


        sentences.extend(batch_sentences.numpy().tolist())
        predictions.extend(batch_preds.numpy().tolist())


    prediction_df = pd.DataFrame({
        'sentence': [s.decode('utf-8') for s in sentences],
        'prediction': predictions
    })

    return prediction_df

gdelt_list = pd.read_csv(r"./Financial sentiment analysis data/gdelt.csv", encoding='unicode_escape').set_index('filename')
gdelt_colName = pd.read_csv(r"./Financial sentiment analysis data/CSV.header.dailyupdates.txt", sep='\t', header=None).iloc[0].to_list()
sentiment_list = ["positive", "neutral", "negative"]
start_time = time.time()

for gdelt_list_counter in range(0, len(gdelt_list)):

    zip_file_url = urlopen(gdelt_list.iloc[gdelt_list_counter]['hyperlink'])
    zip_file = ZipFile(BytesIO(zip_file_url.read()))
    document_list = pd.read_csv(zip_file.open(zip_file.namelist()[0]), sep='\t', header=None,
                                names=gdelt_colName).set_index(['GLOBALEVENTID'])
    US_document_list = document_list[
        (document_list['Actor1Geo_ADM1Code'] == "US") & (document_list['Actor2Geo_ADM1Code'] == "US") & (
                    document_list['ActionGeo_ADM1Code'] == "US") & (
                    (document_list['Actor1Code'] == "USA") | (document_list['Actor2Code'] == "USA"))]

    text_list = []
    exception_count = 0
    for doc_count in range(0, len(US_document_list)):

        if doc_count % READ_FREQ == 0 or exception_count == 1:
            news_url = US_document_list.iloc[doc_count]['SOURCEURL']
            print(f"Processing doc {doc_count + 1}/{len(US_document_list)}: {news_url}")  # 详细日志

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Connection": "keep-alive"
            }

            result = fetch_url_with_timeout(
                url=news_url,
                headers=headers,
                timeout=30
            )

            if result['error']:
                print(f"Error processing URL {news_url}: {result['error']}")
                exception_count = 1
            else:
                text_list.append({'sentence': result['text']})
                exception_count = 0

        current_time = time.time()
        sys.stdout.write("\rgdelt_list_counter = %s / %s ; doc_count = %s / %s ; time elasped: %s minutes." % (
        gdelt_list_counter + 1, len(gdelt_list), doc_count + 1, len(US_document_list),
        round((current_time - start_time) / 60, 2)))
        sys.stdout.flush()

    filename = zip_file.namelist()[0]
    news_year = int(filename[0:4])
    news_month = int(filename[4:6])
    news_day = int(filename[6:8])

    text_matrix = pd.DataFrame(text_list)
    text_matrix['sentiment_label'] = 0
    try:
        text_matrix['sentiment_svc'] = svc_model.predict(text_matrix['sentence'])
        text_matrix['sentiment_trans'] = \
        predict_unseen_test(transformer_model, create_dataset(text_matrix, bert_tokenizer, MAX_LEN, BATCH_SIZE))['prediction']

        # sentiment_score_svc.loc[(news_year, news_month, news_day), 'exception'] += exception_count
        # sentiment_score_trans.loc[(news_year, news_month, news_day), 'exception'] += exception_count
        for i in sentiment_list:
            sentiment_score_svc.loc[(news_year, news_month, news_day), i] += sum(text_matrix['sentiment_svc'] == i)
            sentiment_score_trans.loc[(news_year, news_month, news_day), i] += sum(
                text_matrix['sentiment_trans'] == sentiment_list.index(i))
    except Exception as e:
        print(f"Error in prediction processing: {str(e)}")
        pass

    del zip_file, document_list, US_document_list, text_matrix
    gc.collect()

    print("\rReading GEDLT progress: %s / %s ; Time Elasped: %s minutes." % (
    gdelt_list_counter + 1, len(gdelt_list), round((current_time - start_time) / 60, 2)))
    if (gdelt_list_counter % SAVING_FREQ == 0):
        sentiment_score_svc.to_csv(r"./sentiment_score_svc.csv")
        sentiment_score_trans.to_csv(r"./sentiment_score_trans.csv")

sentiment_score_svc.to_csv(r"./sentiment_score_svc.csv")
sentiment_score_trans.to_csv(r"./sentiment_score_trans.csv")


