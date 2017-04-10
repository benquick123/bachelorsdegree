from datetime import datetime

import pymongo
import pytz
import numpy as np
import other.textrank as textrank
from other.rake import Rake
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import MeanShift
import pickle

from operator import itemgetter


def delete_duplicates(a):
    to_delete = [True]
    for i in range(len(a)-1):
        if a[i]["date"] == a[i+1]["date"] and a[i]["text"] == a[i+1]["text"]:
            to_delete.append(False)
        else:
            to_delete.append(True)

    if len(to_delete) > 0:
        last_true = 0
        for i, delete in enumerate(to_delete):
            if delete:
                last_true = i
            else:
                a[last_true]["currencies"].append(a[i]["currencies"][0])

        a = a[np.array(to_delete)]

    return a


def load_articles(db_news, currencies, date_from, date_to, p=False):
    articles = []

    for article in db_news.articles.find({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lte": date_to}}, {"$or": [{"relevant": True}, {"relevant_predict": True}]}]}, {"_id": 1, "title": 1, "date": 1, "source": 1, "authors": 1, "text": 1, "stemmed_title": 1, "stemmed_text": 1, "keywords_textrank": 1}):
        currency = article["_id"].split(":")[0]
        if (currency in currencies or len(currencies) == 0) and len(article["title"]) > 0 and len(article["text"]) > 0:
            article["currencies"] = [currency]
            articles.append(article)

    articles = sorted(articles, key=itemgetter("date"))
    articles = delete_duplicates(np.array(articles))

    if p:
        for article in articles:
            print(article["currencies"], article["title"], article["_id"])

    return articles


def get_keywords(articles, db=None):
    # run this only once:
    # textrank.setup_environment()
    rake = Rake("../other/stoplists/FoxStoplist.txt")
    _keywords = []
    for article in articles:
        _textrank = sorted(list(textrank.extract_key_phrases(article["text"].lower())))

        _rake = rake.run(article["text"].lower())
        _rake = sorted([phrase for phrase, _ in _rake])

        if db is not None:
            db.articles.update({"_id": article["_id"]}, {"$set": {"keywords_rake": _rake, "keywords_textrank": _textrank}})

        _keywords.append(_textrank)

    return _keywords


def calc_distances(articles):
    def analyzer(text):
        return text

    keywords = []
    articles = np.array(articles)

    for article in articles:
        keywords.append(article["keywords_textrank"])

    tfidf = TfidfVectorizer(analyzer=analyzer, min_df=0.0020)
    tfidf.fit_transform(keywords)

    # scores = sorted(zip(tfidf.get_feature_names(), tfidf.idf_), key=itemgetter(1))
    # scores = scores[-100:]
    # print(scores)

    count = CountVectorizer(analyzer=analyzer, vocabulary=tfidf.vocabulary_)
    X = count.fit_transform(keywords).todense()

    _X = []
    to_delete = np.zeros(X.shape[0], dtype=bool)

    for i in range(X.shape[0]):
        x = np.array(X[i])[0]
        if np.any(x > 0):
            _X.append(x)
        else:
            to_delete[i] = True

    articles = articles[~to_delete]
    X = np.array(_X)

    try:
        clusters = pickle.load(open("mean_shift_clusters.p", "rb"))
        mean_shift = pickle.load(open("mean_shift.p", "rb"))
        print("loaded from disk")
    except FileNotFoundError:
        mean_shift = MeanShift(n_jobs=1, cluster_all=False)
        clusters = mean_shift.fit_predict(X)

        pickle.dump(clusters, open("mean_shift_clusters.p", "wb"))
        pickle.dump(mean_shift, open("mean_shift.p", "wb"))

    print(X.shape)
    print(len(mean_shift.cluster_centers_))
    return X, clusters, articles


def __main__():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db_news = client["news"]
    currencies = set()

    date_from = datetime(2015, 11, 1, 0, 0, 0).replace(tzinfo=pytz.utc)
    date_to = datetime(2016, 10, 31, 23, 59, 59).replace(tzinfo=pytz.utc)

    articles = load_articles(db_news, currencies, date_from, date_to)

    # call this to get keywords. not necessary since keywords are already stored and loaded from db.
    # get_keywords(articles)

    X, clusters, articles = calc_distances(articles)
    print(clusters.tolist())

__main__()


"""

def get_clusters(articles, plot=False):
    stemmed_texts = []
    for article in articles:
        stemmed_texts.append(" ".join(article.stemmed_text))

    tfidf = TfidfVectorizer(analyzer="word", min_df=0.2)
    X = tfidf.fit_transform(stemmed_texts)

    scores = sorted(zip(tfidf.get_feature_names(), tfidf.idf_), key=itemgetter(1))
    scores = scores[-1000:]
    print(scores)

    col = [tfidf.vocabulary_[word] for word, score in scores]
    X = X[:, col].todense()

    centroids, _ = kmeans(X, 2)
    idx, _ = vq(X, centroids)

    curr = {0: "", 1: "", 2: "", "ZEC": -1, "ETH": -1, "XMR": -1}
    t = 0
    f = 0

    counts = {"ZEC": 0, "ETH": 0, "XMR": 0}
    _counts = {0: 0, 1: 0, 2: 0}

    for i, _id in enumerate(idx):
        if curr[articles[i].currency] == -1:
            curr[articles[i].currency] = _id
            curr[_id] = articles[i].currency

        if curr[articles[i].currency] == _id:
            t += 1
        else:
            f += 1

        counts[articles[i].currency] += 1
        _counts[_id] += 1

    print(t / len(articles))
    print(f / len(articles))
    print(len(articles))
    print()
    print(counts)
    print(_counts)
    #L = fastcluster.linkage(X, "average")
    #c, coph_dist = sch.cophenet(L, pdist(X))

    if plot:
        plt.plot(X[idx == 0, 0], X[idx == 0, 1], 'ob', X[idx == 1, 0], X[idx == 1, 1], 'or')
        plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=1)
        plt.show()


"""