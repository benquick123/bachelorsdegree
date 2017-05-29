import pymongo
import re
import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


def strip_text(db):
    for tweet in db.messages.find():
        text = tweet["text"]
        striped_text = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
        db.messages.update_one({"_id": tweet["_id"]}, {"$set": {"striped_text": striped_text}})


def remove_non_english(client):
    db_twitter = client["twitter"]
    db_prices = client["crypto_prices"]

    prices = set(db_prices.collection_names())
    with open("../other/wordsEnDictionary.txt") as word_file:
        words = set(word.replace("\n", "") for word in word_file)

    for tweet in db_twitter.messages.find():
        if tweet["crypto_currency"].lower() in prices:
            striped_text = tweet["striped_text"]
            new_text = " ".join([word for word in nltk.wordpunct_tokenize(striped_text) if word.lower() in prices or word.lower() in words])
            print(tweet["_id"], "\n", striped_text, "\n\t", new_text, "\n", sep="")

            db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"striped_text": new_text}})
            if len(new_text) > 10:
                db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"relevant": True}})
            else:
                db_twitter.messages.upžigadatecesar_one({"_id": tweet["_id"]}, {"$set": {"relevant": False}})
        else:
            db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"relevant": False}})


def classification(db, k):
    n = db.messages.count({"relevant_classifier": {"$exists": False}, "relevant": True})
    k = k / n
    selection = np.random.choice(2, n, p=[1-k, k])

    i = 0
    for tweet in db.messages.find({"relevant_classifier": {"$exists": False}, "relevant": True}, no_cursor_timeout=True):
        if selection[i] == 1:
            print(tweet["text"], "\n")
            print(tweet["striped_text"])
            ans = input(tweet["_id"] + ": Relevant? ")
            if ans == "1":
                db.messages.update_one({"_id": tweet["_id"]}, {"$set": {"relevant_classifier": True}})
            else:
                db.messages.update_one({"_id": tweet["_id"]}, {"$set": {"relevant_classifier": False}})
        i += 1


def load_tweets(db):
    tweets = []
    for tweet in db.messages.find({"relevant": True, "relevant_classifier": {"$exists": True}}, {"_id": 1, "striped_text": 1, "relevant_classifier": 1, "text": 1}):
        tweets.append(tweet)
    return tweets


def load_tweets_no_predictions(db):
    tweets = []
    for tweet in db.messages.find({"relevant": True, "relevant_classifier": {"$exists": False}}, {"_id": 1, "relevant_classifier": 1, "striped_text": 1, "text": 1}):
        tweets.append(tweet)
    return tweets


def format_data(tweets, vocabulary=None, predictions=False):
    Y = []
    striped_texts = []
    for tweet in tweets:
        striped_texts.append(tweet["text"])
        if predictions:
            Y.append(tweet["relevant_classifier"])

    if vocabulary is not None:
        count = CountVectorizer(analyzer="word", vocabulary=vocabulary)
    else:
        count = CountVectorizer(analyzer="word")
    X = count.fit_transform(striped_texts)

    if predictions:
        return count, X.todense(), Y
    else:
        return count, X.todense()


def predict_all(db, model, tweets, vocabulary, write=False):
    begin = 0
    end = 1000

    while end <= len(tweets):
        _tweets = tweets[begin:end]
        counts, X = format_data(tweets=_tweets, vocabulary=vocabulary, predictions=False)

        Y = model.predict(X)
        print(len(np.where(Y)[0]), "/", len(Y))

        if write:
            for tweet, prediction in zip(_tweets, Y):
                p = True if prediction else False
                db.messages.update({"_id": tweet["_id"]}, {"$set": {"relevant_predict": p}})

        begin = end
        end += 1000

    end = len(tweets)
    _tweets = tweets[begin:end]
    counts, X = format_data(_tweets, vocabulary=vocabulary, predictions=False)

    Y = model.predict(X)
    print(len(np.where(np.array(Y) is True)[0]), "/", len(Y))

    if write:
        for tweet, prediction in zip(_tweets, Y):
            p = True if prediction else False
            db.messages.update({"_id": tweet["_id"]}, {"$set": {"relevant_predict": p}})


def __main__():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["twitter"]

    # strip_text(db)
    # remove_non_english(client)
    # classification(db, 2000)

    tweets = load_tweets(db)
    counts, X, Y = format_data(tweets, vocabulary=None, predictions=True)

    model = LinearSVC()
    scores = cross_val_score(model, X, Y, cv=50)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    input("Continue?")
    model.fit(X, Y)
    _tweets = np.array(load_tweets_no_predictions(db))
    predict_all(db, model, _tweets, counts.vocabulary_, write=True)

__main__()
