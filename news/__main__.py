from datetime import datetime
from other.Article import Article

import pymongo
import pytz
import numpy as np
import other.textrank as textrank
from other.rake import Rake
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import MeanShift
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

from operator import itemgetter
import time


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

    for article in db_news.articles.find({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lte": date_to}}, {"$or": [{"relevant": True}, {"relevant_predict": True}]}]}, {"_id": 1, "title": 1, "date": 1, "source": 1, "authors": 1, "text": 1, "stemmed_title": 1, "stemmed_text": 1, "keywords_textrank": 1, "sentiment_text": 1, "sentiment_title": 1}):
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

    # print(X.shape)
    print("n clusters:", len(mean_shift.cluster_centers_))
    return X, clusters, articles


def create_articles(client, articles, clusters, threshold=0.0):
    _articles = []
    for i, article in enumerate(articles):
        time_start = int(time.mktime(article["date"].timetuple()) + 3600)
        time_end = time_start + 6*3600

        _article = Article(article["_id"], date=time_start, title=article["title"], text=article["text"], authors=article["authors"], source=article["source"])
        _article.stemmed_text = article["stemmed_text"]
        _article.stemmed_title = article["stemmed_title"]
        _article.cluster = clusters[i]

        currencies = set(article["currencies"])
        currencies_to_del = set()

        for currency in currencies:
            pmin = 99999999
            pmax = -99999999
            tmin = 0
            tmax = 0

            first_price = 0

            for price in client["crypto_prices"][currency.lower()].find({"$and": [{"_id": {"$gte": time_start}}, {"_id": {"$lte": time_end}}]}):
                if first_price == 0:
                    first_price = price["weightedAverage"]
                else:
                    if price["weightedAverage"] < pmin:
                        pmin = price["weightedAverage"]
                        tmin = price["_id"] - time_start
                    if price["weightedAverage"] > pmax:
                        pmax = price["weightedAverage"]
                        tmax = price["_id"] - time_start

            if first_price > 0:
                pmin = (pmin - first_price) / first_price
                pmax = (pmax - first_price) / first_price

                if abs(pmin) >= threshold or abs(pmax) >= threshold:
                    _article.pmin.append(pmin)
                    _article.pmax.append(pmax)
                    _article.tmin.append(tmin)
                    _article.tmax.append(tmax)

                    if abs(pmin) >= abs(pmax):
                        _article.classification.append(0)
                    else:
                        _article.classification.append(1)
                else:
                    currencies_to_del.add(currency)
            else:
                currencies_to_del.add(currency)

        if len(currencies) != len(currencies_to_del):
            if "sentiment_text" in article.keys():
                _article.sentiment_title = article["sentiment_title"]
                _article.sentiment_text = article["sentiment_text"]
            else:
                _article = sentiment_analyzer(_article, db=client["news"])

            currencies -= currencies_to_del
            _article.currencies = currencies
            _article.map_currencies()
            _articles.append(_article)

    return _articles


def sentiment_analyzer(article, db=None):
    analyzer = SentimentIntensityAnalyzer()
    scores_text = analyzer.polarity_scores(article.text)
    scores_title = analyzer.polarity_scores(article.title)

    if db is not None:
        print("calculating and saving sentiment")
        db.articles.update({"_id": article.id}, {"$set": {"sentiment_text": scores_text, "sentiment_title": scores_title}})
    article.sentiment_text = scores_text
    article.sentiment_title = scores_title
    return article


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

    # price change threshold: [0:1]
    articles = create_articles(client, articles, clusters, threshold=0.1)
    print("n articles:", len(articles))

    # currency, text_pos, text_neu, text_neg, text_comb, title_pos, title_neu, title_neg, title_comb, classification
    x = []
    for article in articles:
        article_matrix = article.get_article_matrix()
        for line in article_matrix:
            x.append(line)

    x = np.array(x)
    X = x[:, :-1]
    Y = x[:, -1]
    model = LinearSVC()
    scores = cross_val_score(model, X, Y, cv=50)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

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