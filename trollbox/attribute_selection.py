import pymongo
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pytz
import numpy as np
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter


def get_term_counts(currency, date_start, date_end):
    corpus = OrderedDict()

    date_start_prepare = int(date_start.timestamp())
    date_end_prepare = int(date_end.timestamp())

    while date_start_prepare < date_end_prepare:
        d = datetime.datetime.utcfromtimestamp(date_start_prepare).strftime('%Y-%m-%d %H')
        corpus[d] = ""
        date_start_prepare += 3600

    date_start = date_start.strftime("%Y-%m-%d %H:%M:%S UTC")
    date_end = date_end.strftime("%Y-%m-%d %H:%M:%S UTC")

    for message in db_chat.messages.find({"coin_mentions": {"$in": currency}, "$and": [{"date": {"$gte": date_start}}, {"date": {"$lte": date_end}}]}, {"_id": 1, "text": 1, "date": 1}).sort("date", 1):
        date_split = message["date"][:-10]
        if len(corpus[date_split]) == 0:
            corpus[date_split] += message["text"]
        else:
            corpus[date_split] += " " + message["text"]

    corpus = list(corpus.values())
    vectorizer = TfidfVectorizer(analyzer="word", min_df=0.10, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer, corpus


def get_crypto_prices(currency, date_start, date_end):
    date_start = int(date_start.timestamp()) + 1 #+1 second to ignore the first hour
    date_end = int(date_end.timestamp()) + 1 #+1 second to get value for last hour

    weigh_avg_prices = []

    for price_info in db_prices[currency].find({"$and": [{"_id": {"$gte": date_start}}, {"_id": {"$lte": date_end}}]}, {"_id": 1, "weightedAverage": 1}).sort("_id", 1):
        date = price_info["_id"]
        if date % 3600 == 0:
            weigh_avg_prices.append(price_info["weightedAverage"])

    return weigh_avg_prices


def get_word_tf(word):
    word_to_int = {value: i for i, value in enumerate(vectorizer.get_feature_names())}
    tf_tmp = []
    tf = []
    # calculate tf for each time period
    # calculate absolute change between time period i and i-1
    for i, n in enumerate(X.toarray()[:, word_to_int[word]]):
        tf_tmp.append(n / len(corpus[i]))
        if i > 0:
            tf.append(tf_tmp[i] - tf_tmp[i - 1])
    return tf


client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db_chat = client["trollbox"]
db_prices = client["crypto_prices"]

date_start = datetime.datetime(2016, 1, 1, 0, 0, 0).replace(tzinfo=pytz.utc)
date_end = datetime.datetime(2016, 2, 28, 23, 59, 59).replace(tzinfo=pytz.utc)
currency = ["btc", "bitcoin"]

X, vectorizer, corpus = get_term_counts(currency, date_start, date_end)

tf_idf_scores = sorted(zip(vectorizer.get_feature_names(), vectorizer.idf_), key=itemgetter(1))
for i in range(0, 50):
    print(tf_idf_scores[-(i+1)])

#EXIT
exit()

weigh_avg_tmp = get_crypto_prices(currency[0], date_start, date_end)
weigh_avg = []
for i, price in enumerate(weigh_avg_tmp):
    if i > 0:
        if weigh_avg_tmp[i-1] == 0:
            weigh_avg.append(0)
        else:
            weigh_avg.append((weigh_avg_tmp[i] - weigh_avg_tmp[i-1]) / weigh_avg_tmp[i-1])

buy_tf = get_word_tf("buy")
sell_tf = get_word_tf("sell")

data = []
n = 3

for i in range(n, len(weigh_avg)):
    sample = []
    for j in range(1, n+1):
        sample.append(buy_tf[i-j])
        sample.append(sell_tf[i-j])
    sample.append(0 if weigh_avg[i] <= 0 else 1)
    data.append(sample)

data = np.array(data)
to_remove = np.zeros(len(data), dtype=bool)
for i in range(len(data[0, :])):
    to_remove = np.logical_or(to_remove, np.isnan(data[:, i]))

data = data[~to_remove, :]

#data = preprocessing.scale(data)
#data = preprocessing.normalize(data)

model = GaussianNB()
X = data[:, :-1]
Y = data[:, -1]
scores = cross_val_score(model, X, Y, cv=50)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
