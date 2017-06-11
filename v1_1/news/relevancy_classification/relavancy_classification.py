import pymongo
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


def load_articles(db):
    articles = []
    for article in db.articles.find({"relevant": {"$exists": 1}}):
        articles.append(article)
    return articles


def load_articles_no_predictions(db):
    articles = []
    for article in db.articles.find({"relevant": {"$exists": 0}}):
        articles.append(article)
    return articles


def format_data(articles, predictions=False, vocabulary=None):
    stemmed_texts = []
    Y = []
    for article in articles:
        stemmed_texts.append(" ".join(article["stemmed_text"]))
        if predictions:
            Y.append(article["relevant"])

    if vocabulary is not None:
        count = CountVectorizer(analyzer="word", vocabulary=vocabulary)
    else:
        count = CountVectorizer(analyzer="word")
    X = count.fit_transform(stemmed_texts)

    if predictions:
        return count, X.todense(), Y
    else:
        return count, X.todense()


def predict_all(model, articles, vocabulary, write=False):
    begin = 0
    end = 1000
    while end <= len(articles):
        _articles = articles[begin:end]
        _counts, _X = format_data(_articles, vocabulary=vocabulary)

        _Y = model.predict(_X)
        print(len(np.where(np.array(Y) is True)[0]), "/", len(Y), ":", len(np.where(np.array(_Y) is True)[0]), "/", len(_Y))
        if write:
            for article, prediction in zip(_articles, _Y):
                _p = True if prediction is True else False
                db.articles.update({"_id": article["_id"]}, {"$set": {"relevant_predict": _p}})

        begin = end
        end += 1000

    end = len(articles)
    _articles = articles[begin:end]
    _counts, _X = format_data(_articles, vocabulary=vocabulary)

    _Y = model.predict(_X)
    print(len(np.where(np.array(Y) is True)[0]), "/", len(Y), ":", len(np.where(np.array(_Y) is True)[0]), "/", len(_Y))
    if write:
        for article, prediction in zip(_articles, _Y):
            _p = True if prediction is True else False
            db.articles.update({"_id": article["_id"]}, {"$set": {"relevant_predict": _p}})


client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["news"]

a = np.array(load_articles(db))
counts, X, Y = format_data(a, True)

model = SGDClassifier()
scores = cross_val_score(model, X, Y, cv=50)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

input("Continue?")
model.fit(X, Y)
_a = np.array(load_articles_no_predictions(db))
predict_all(model, _a, counts.vocabulary_, write=True)
