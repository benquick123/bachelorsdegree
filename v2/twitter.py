import pymongo
import v2.common as common
import re


def load_with_attr():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["twitter"]

    tweets = []
    for tweet in db.messages.find({"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}, {"_id": 1, "text": 1, "retweet_count": 1, "posted_time": 1, "favorites_count": 1, "author": 1, "crypto_currency": 1, "sentiment": 1, "polartiy": 1, "clean_text": 1}):
        if len(tweet["clean_text"]) > 1:
            tweets.append(tweet)

    return tweets


def load_without_attr(save_to_db=True):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["twitter"]

    pattern = re.compile("[\W_]+")

    tweets = []
    for tweet in db.messages.find({"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}, {"_id": 1, "text": 1, "retweet_count": 1, "posted_time": 1, "favorites_count": 1, "author": 1, "crypto_currency": 1}):
        text = []
        for word in tweet["text"].split(" "):
            word = word.lower()
            word = remove_hyperlinks(word)
            if word is None:
                continue

            word = common.remove_special_chars(word, pattern)
            if word is None:
                continue

            word = remove_non_ascii_chars(word)
            if word is None:
                continue

            if not common.is_stop_word(word):
                text.append(word)

        sentiment, polarity = common.calc_sentiment_polarity(text)
        text = common.stem(text)

        tweet["clean_text"] = text
        tweet["sentiment"] = sentiment
        tweet["polarity"] = polarity

        if save_to_db:
            db.messages.update_one({"_id": tweet["_id"]}, {"$set": {"clean_text": text, "sentiment": sentiment, "polarity": polarity}})

        if len(text) > 1:
            tweets.append(tweet)

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


def remove_non_ascii_chars(word):
    word = word.decode("utf-8").encode("ascii", errors="ignore")
    if len(word) == 0:
        return None
    return word


def create_matrix(tweets):
    X = []
    Y = []
    for tweet in tweets:
        _X, _Y = create_matrix_line(tweet)
        X.append(_X)
        Y.append(_Y)
    return X, Y


def create_matrix_line(tweet_data):
    _X = []
    _Y = 0
    return _X, _Y
