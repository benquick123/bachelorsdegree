import pymongo
import v2.common as common
import re
import string
import time
from scipy import sparse
import numpy as np


def load_with_attr(n=None, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["twitter"]

    tweets = []
    i = 0
    for tweet in db.messages.find({"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}, {"_id": 1, "text": 1, "retweet_count": 1, "posted_time": 1, "favorites_count": 1, "author": 1, "crypto_currency": 1, "sentiment": 1, "polarity": 1, "clean_text": 1, "pos_tag_text": 1}, no_cursor_timeout=True).sort("posted_time", 1):
        if n is None or i < n:
            if p:
                print("processing tweet", i, "(" + str(tweet["_id"]) + ")")

            if len(tweet["clean_text"]) > 1:
                tweets.append(tweet)
            i += 1
        else:
            break

    topics = common.generate_topics(tweets, "clean_text", 50)
    for tweet, topic_attrs in zip(tweets, topics):
        tweet["topics"] = topic_attrs

    return tweets


def load_without_attr(p=False, save_to_db=True):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["twitter"]

    pattern = re.compile("[\W]+")
    printable = set(string.printable)

    tweets = []
    i = 0
    for tweet in db.messages.find({"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}, {"_id": 1, "text": 1, "retweet_count": 1, "posted_time": 1, "favorites_count": 1, "author": 1, "crypto_currency": 1}, no_cursor_timeout=True).sort("posted_time", 1):
        if p:
            print("processing tweet", i, "(" + str(tweet["_id"]) + ")")
            i += 1

        text = []
        for word in tweet["text"].split(" "):
            word = word.lower()
            word = remove_hyperlinks(word)
            if word is None:
                continue

            word = common.remove_special_chars(word, pattern)
            if word is None:
                continue

            word = remove_non_ascii_chars(word, printable)
            if word is None:
                continue

            if not common.is_stop_word(word) or word == "not":
                text.append(word)

        pos_tag_text = common.get_pos_tags(text)
        sentiment, polarity = common.calc_sentiment_polarity(text, pos_tag_text)
        text = common.stem(text)

        tweet["clean_text"] = text
        tweet["pos_tag_text"] = pos_tag_text
        tweet["sentiment"] = sentiment
        tweet["polarity"] = polarity

        if save_to_db:
            db.messages.update_one({"_id": tweet["_id"]}, {"$set": {"clean_text": text, "sentiment": sentiment, "polarity": polarity, "pos_tag_text": pos_tag_text}})

        if len(text) > 1:
            tweets.append(tweet)

    topics = common.generate_topics(tweets, "clean_text", 100)
    for tweet, topic_attrs in zip(tweets, topics):
        tweet["topics"] = topic_attrs

    return tweets


def remove_hyperlinks(word):
    if "t.c" in word:
        return None
    if "https" in word:
        return None
    if "http" in word:
        return None
    if "www." in word:
        return None
    return word


def remove_non_ascii_chars(word, printable):
    _word = "".join(char for char in word if char in printable)
    if len(_word) == 0:
        return None
    return _word


def get_user_info(client, tweet):
    db = client["twitter"]
    user = db.users.find_one({"_id": tweet["author"]}, {"followers_count": 1, "verified": 1, "_id": 0})
    return [user["followers_count"], 1 if user["verified"] else 0]


def create_matrix(tweets, window, margin, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    X = None
    Y = []

    tfidf, vocabulary = common.calc_tf_idf(tweets, "clean_text")
    financial_weights = common.create_financial_vocabulary(vocabulary)

    for tweet, tweet_tfidf in zip(tweets, tfidf):
        if p:
            print("processing tweet", str(tweet["_id"]), "(" + tweet["crypto_currency"] + ")", )

        _X, _Y = create_matrix_line(client, tweet, tweet_tfidf, financial_weights, window, margin)
        if _Y is not None and _X is not None:
            if X is None:
                X = sparse.csr_matrix(_X)
            else:
                X = sparse.vstack([X, _X])
            Y.append(_Y)

    return X, Y


def create_matrix_line(client, tweet_data, tfidf, financial_weights, window, margin):
    averaged_results = common.get_averaged_results(client, tweet_data["posted_time"], 24*3600, tweet_data["crypto_currency"])
    user_info = get_user_info(client, tweet_data)
    _X = [tweet_data["sentiment"], tweet_data["polarity"]] + user_info + averaged_results + tweet_data["topics"]

    if np.any(np.isnan(_X)) or np.any(np.isinf(_X)):
        return None, None

    tfidf = tfidf.todense()
    tfidf = np.where(tfidf != 0, tfidf + financial_weights, 0).tolist()[0]
    _X += tfidf

    date_from = int(time.mktime(tweet_data["posted_time"].timetuple()) + 3600)
    date_to = date_from + window
    percent_change = common.get_price_change(client, tweet_data["crypto_currency"].lower(), date_from, date_to)
    if percent_change is not None:
        _Y = 0
        if percent_change > margin:
            _Y = 1
        elif percent_change < -margin:
            _Y = -1
    else:
        return None, None

    return sparse.csr_matrix(_X), _Y
