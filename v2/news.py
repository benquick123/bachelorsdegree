import pymongo
import v2.common as common
import re
import time
from scipy import sparse
import numpy as np
from datetime import datetime

articles = []


def load_with_attr(n=None, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["news"]

    i = 0
    for article in db.articles.find({"relevant": True}, {"_id": 1, "title": 1, "text": 1, "date": 1, "source": 1, "authors": 1, "currency": 1, "clean_title": 1, "clean_text": 1, "title_sentiment": 1, "title_polarity": 1, "text_sentiment": 1, "text_polarity": 1, "topic": 1, "pos_tag_text": 1, "pos_tag_title": 1}, no_cursor_timeout=True).sort("date", 1):
        if n is None or i < n:
            if p:
                print("processing article", i, "(" + str(article["_id"]) + ")")

            if len(article["clean_text"]) > 1 and len(article["clean_title"]) > 1 and (article["date"].hour != 0 or article["date"].minute != 0 or article["date"].second != 0):
                articles.append(article)
            i += 1
        else:
            break

    topics = common.generate_topics(articles, "clean_text", 100)
    for article, topic_attrs in zip(articles, topics):
        article["topics"] = topic_attrs

    return articles


def load_without_attr(p=False, save_to_db=True):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["news"]

    pattern = re.compile("[\W]+")

    # articles = []
    i = 0
    for article in db.articles.find({"relevant": True}, {"_id": 1, "title": 1, "text": 1, "date": 1, "source": 1, "authors": 1, "currency": 1}, no_cursor_timeout=True).sort("date", 1):
        if p:
            print("processing article", i, "(" + str(article["_id"]) + ")")
            i += 1
        title = []
        text = []

        for word in article["title"].split(" "):
            word = word.lower()

            word = common.remove_special_chars(word, pattern)
            if word is None:
                continue

            # if not common.is_stop_word(word) or word == "not":
            title.append(word)

        for word in article["text"].split(" "):
            word = word.lower()

            word = common.remove_special_chars(word, pattern)
            if word is None:
                continue

            # if not common.is_stop_word(word) or word == "not":
            text.append(word)

        pos_tag_title = common.get_pos_tags(title)
        pos_tag_text = common.get_pos_tags(text)
        title = common.lemmatize(title, pos_tag_title)
        text = common.lemmatize(text, pos_tag_text)

        title_sentiment, title_polarity = common.calc_sentiment_polarity(title)
        text_sentiment, text_polarity = common.calc_sentiment_polarity(text)

        article["clean_title"] = title
        article["clean_text"] = text
        article["title_sentiment"] = title_sentiment
        article["text_sentiment"] = text_sentiment
        article["title_polarity"] = title_polarity
        article["text_polarity"] = text_polarity
        article["pos_tag_title"] = pos_tag_title
        article["pos_tag_text"] = pos_tag_text

        if save_to_db:
            db.articles.update_one({"_id": article["_id"]}, {"$set": {"clean_title": title, "clean_text": text, "title_sentiment": title_sentiment, "title_polarity": title_polarity, "text_sentiment": text_sentiment, "text_polarity": text_polarity, "pos_tag_text": pos_tag_text, "pos_tag_title": pos_tag_title}})

        if len(title) > 1 and len(text) > 1:
            pass
            # articles.append(article)

    """topics = common.generate_topics(articles, "clean_text", 100)
    for article, topic_attrs in zip(articles, topics):
        article["topics"] = topic_attrs"""

    return None     # articles


def create_matrix(articles, window, margin, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    X = None
    Y = []

    tfidf, vocabulary = common.calc_tf_idf(articles, min_df=0.0015, max_df=1.0, dict_key="clean_text")
    financial_weights = common.create_financial_vocabulary(vocabulary) + 1
    sentiment_weights = common.calc_sentiment_for_vocabulary(vocabulary)
    sentiment_weights = np.where(sentiment_weights >= 0, sentiment_weights + 1, sentiment_weights - 1)
    weights = sparse.csr_matrix(financial_weights * sentiment_weights)

    """for word in zip(vocabulary, financial_weights.tolist(), sentiment_weights):
        print(word)"""

    date_start = datetime(2015, 11, 2, 0, 0, 0)

    for i, (article, article_tfidf) in enumerate(zip(articles, tfidf)):
        if p:
            print("processing article", str(article["_id"]), "(" + article["currency"] + ")", )

        article["tfidf"] = article_tfidf
        if article["date"] >= date_start:
            _X, _Y = create_matrix_line(client, i, article, weights, window, margin)
            if _Y is not None and _X is not None:
                if X is None:
                    X = sparse.csr_matrix(_X)
                else:
                    X = sparse.vstack([X, _X])
                Y.append(_Y)

    return X, Y


def create_matrix_line(client, i, article_data, weights, window, margin):
    date_from = int(time.mktime(article_data["date"].timetuple()) + 3600)
    date_to = date_from + window
    percent_change = common.get_price_change(client, article_data["currency"].lower(), date_from, date_to)

    if not np.isnan(percent_change):
        _Y = 0
        if percent_change > margin:
            _Y = 1
        elif percent_change <= -margin:
            _Y = -1
    else:
        return None, None

    _price_change = common.get_price_change(client, article_data["currency"].lower(), date_from-(24*3600), date_from)

    db_averages = common.get_averages_from_db(client, article_data["date"], 24*3600, article_data["currency"], articles=False)
    data_averages, average_tfidf, n, average_topics = common.get_averages_from_data(articles, article_data["date"], 24*3600, article_data["currency"], i, 0.004, type="article")

    topics = (n / (n+1)) * average_topics + (1 / (n+1)) * article_data["topics"]

    _X = [1 / (n+1), article_data["title_sentiment"], article_data["text_sentiment"], article_data["title_polarity"], article_data["text_polarity"], _price_change] + data_averages + db_averages + list(topics)

    if not np.all(np.isfinite(_X)):
        return None, None

    tfidf = article_data["tfidf"] * (1 / (n + 1)) + average_tfidf * (n / (n + 1))
    tfidf = tfidf.multiply(weights)
    _X += tfidf.todense().tolist()[0]

    return sparse.csr_matrix(_X), _Y

