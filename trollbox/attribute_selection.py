import pymongo
import datetime
from sklearn.feature_extraction.text import CountVectorizer
import pytz
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def get_term_counts(currency, date_start, date_end):
    corpus = []
    tmp_corpus = []

    date_start = date_start.strftime("%Y-%m-%d %H:%M:%S UTC")
    date_end = date_end.strftime("%Y-%m-%d %H:%M:%S UTC")
    last_date = date_start

    for message in db_chat.messages.find({"coin_mentions": {"$in": currency}, "$and": [{"date": {"$gte": date_start}}, {"date": {"$lte": date_end}}]}, {"_id": 1, "text": 1, "date": 1}).sort("date", 1):
        date_split = message["date"].split(" ")
        date = date_split[0]
        time = date_split[1]

        hour = int(time.split(":")[0])
        if date != last_date:
            last_date = date
            corpus += tmp_corpus
            tmp_corpus = []

        if len(tmp_corpus) <= hour:
            tmp_corpus.append(message["text"])
        else:
            tmp_corpus[hour] += " " + message["text"]

    corpus += tmp_corpus

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names(), corpus


def get_crypto_prices(currency, date_start, date_end):
    date_start = int(date_start.timestamp()) + 1 #+1 second to ignore the first hour
    date_end = int(date_end.timestamp()) + 1 #+1 second to get value for last hour

    weigh_avg_prices = []

    for price_info in db_prices[currency].find({"$and": [{"_id": {"$gte": date_start}}, {"_id": {"$lte": date_end}}]}, {"_id": 1, "weightedAverage": 1}).sort("_id", 1):
        date = price_info["_id"]
        if date % 3600 == 0:
            weigh_avg_prices.append(price_info["weightedAverage"])

    return weigh_avg_prices


client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db_chat = client["trollbox"]
db_prices = client["crypto_prices"]

date_start = datetime.datetime(2016, 8, 1, 0, 0, 0).replace(tzinfo=pytz.utc)
date_end = datetime.datetime(2016, 9, 30, 23, 59, 59).replace(tzinfo=pytz.utc)
currency = ["btc", "bitcoin"]

X, feature_names, corpus = get_term_counts(currency, date_start, date_end)

word_to_int = {value: i for i, value in enumerate(feature_names)}
buy_tf_tmp = []
sell_tf_tmp = []
buy_tf = []
sell_tf = []

#calculate tf for each time period
#calculate absolute change between time period i and i-1
for i, n in enumerate(X.toarray()[:, word_to_int["buy"]]):
    buy_tf_tmp.append(n / len(corpus[i]))
    if i > 0:
        #buy_tf.append(buy_tf_tmp[i])
        buy_tf.append(buy_tf_tmp[i] - buy_tf_tmp[i-1])

for i, n in enumerate(X.toarray()[:, word_to_int["sell"]]):
    sell_tf_tmp.append(n / len(corpus[i]))
    if i > 0:
        #sell_tf.append(sell_tf_tmp[i])
        sell_tf.append(sell_tf_tmp[i] - sell_tf_tmp[i-1])

weigh_avg_tmp = get_crypto_prices(currency[0], date_start, date_end)
weigh_avg = []
for i, price in enumerate(weigh_avg_tmp):
    if i > 0:
        weigh_avg.append((weigh_avg_tmp[i] - weigh_avg_tmp[i-1]) / weigh_avg_tmp[i-1])

data = []
n = 3
for i in range(n, len(weigh_avg)):
    sample = []
    for j in range(1, n+1):
        sample.append(buy_tf[i-j])
        sample.append(sell_tf[i-j])
    sample.append(weigh_avg[i])
    data.append(sample)

data = np.array(data)
data = preprocessing.scale(data)
data = preprocessing.normalize(data)

model = Ridge()
print(data)
X = data[:, :-1]
Y = data[:, -1]
scores = cross_val_score(model, X, Y, cv=50)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
