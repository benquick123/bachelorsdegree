import pymongo
import common as common
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
    for article in db.articles.find({"relevant": True}, {"_id": 1, "title": 1, "text": 1, "date": 1, "source": 1, "authors": 1, "currency": 1, "clean_title": 1, "clean_text": 1, "title_sentiment": 1, "title_polarity": 1, "text_sentiment": 1, "text_polarity": 1, "topic": 1, "pos_tag_text": 1, "pos_tag_title": 1, "reduced_text": 1, "reduced_text_sentiment": 1, "reduced_text_polarity": 1, "currency_in_title": 1}, no_cursor_timeout=True).sort("date", 1):
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

        indexes = find_currency_occurence(text, article["currency"], common.currencies[article["currency"].lower()])
        reduced_text = extract_n_words_around(text, indexes, n=50)
        reduced_text_sentiment, reduced_text_polarity = common.calc_sentiment_polarity(reduced_text)

        in_title = 1 if is_in_title(title, article["currency"], common.currencies[article["currency"].lower()]) else 0

        article["clean_title"] = title
        article["clean_text"] = text
        article["title_sentiment"] = title_sentiment
        article["text_sentiment"] = text_sentiment
        article["title_polarity"] = title_polarity
        article["text_polarity"] = text_polarity
        article["pos_tag_title"] = pos_tag_title
        article["pos_tag_text"] = pos_tag_text
        article["reduced_text"] = reduced_text
        article["reduced_text_sentiment"] = reduced_text_sentiment
        article["reduced_text_polarity"] = reduced_text_polarity
        article["currency_in_title"] = in_title

        if save_to_db:
            db.articles.update_one({"_id": article["_id"]}, {"$set": {"clean_title": title, "clean_text": text, "title_sentiment": title_sentiment, "title_polarity": title_polarity, "text_sentiment": text_sentiment, "text_polarity": text_polarity, "pos_tag_text": pos_tag_text, "pos_tag_title": pos_tag_title, "reduced_text": reduced_text, "reduced_text_sentiment": reduced_text_sentiment, "reduced_text_polarity": reduced_text_polarity, "currency_in_title": in_title}})

        if len(title) > 1 and len(text) > 1:
            pass
            # articles.append(article)

    """topics = common.generate_topics(articles, "clean_text", 100)
    for article, topic_attrs in zip(articles, topics):
        article["topics"] = topic_attrs"""

    return None     # articles


def find_currency_occurence(text, currency, currency_long):
    indexes = []
    for i, word in enumerate(text):
        word = word.split(".")[0].lower()
        if word == currency.lower() or word == currency_long.lower():
            indexes.append(i)

    return indexes


def extract_n_words_around(text, indexes, n):
    word_indexes_set = set()
    for index in indexes:
        if len(text) < n:
            word_indexes_set = word_indexes_set.union(range(0, len(text)))
            break
        elif index-n/2 >= 0 and index+n/2 < len(text):
            word_indexes_set = word_indexes_set.union(range(int(index-n/2), int(index+n/2+1)))
        elif index-n/2 < 0:
            word_indexes_set = word_indexes_set.union(range(0, n+1))
        elif index+n/2 >= len(text):
            word_indexes_set = word_indexes_set.union(range(len(text)-n, len(text)))

    word_indexes = sorted(list(word_indexes_set))
    return np.array(text)[word_indexes].tolist()


def is_in_title(title, currency, currency_long):
    for word in title:
        word = word.split(".")[0].lower()
        if word == currency.lower() or word == currency_long.lower():
            return True
    return False


def create_matrix(articles, window, margin, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    X = None
    Y = []
    IDs = []

    tfidf, vocabulary = common.calc_tf_idf(articles, min_df=0.0, max_df=1.0, dict_key="reduced_text")
    financial_weights = common.create_financial_vocabulary(vocabulary) + 1
    sentiment_weights = common.calc_sentiment_for_vocabulary(vocabulary)
    sentiment_weights = np.where(sentiment_weights >= 0, sentiment_weights + 1, sentiment_weights - 1)
    weights = sparse.csr_matrix(financial_weights * sentiment_weights)

    """for word in zip(vocabulary, financial_weights.tolist(), sentiment_weights):
        print(word)"""

    date_start = datetime(2015, 11, 2, 0, 0, 0)

    for i, (article, article_tfidf) in enumerate(zip(articles, tfidf)):
        if p:
            print("processing article", str(article["_id"]), "(" + article["currency"] + ")", i)

        article["tfidf"] = article_tfidf
        if article["date"] >= date_start:
            _Y = create_Y(client, article, window, margin)
            if _Y is None:
                continue

            _X = create_X(client, i, article, weights)
            if _X is not None:
                if X is None:
                    X = sparse.csr_matrix(_X)
                else:
                    X = sparse.vstack([X, _X])
                Y.append(_Y)
                IDs.append(article["_id"])

    # general attr
    labels = ["w", "title_sentiment", "reduced_text_sentiment", "title_polarity", "reduced_text_polarity", "curr_in_title"]
    # data_averages
    labels += ["distribution_a_15min", "polarity_a_15min", "sentiment_a_15min"] + ["distribution_a_30min", "polarity_a_30min", "sentiment_a_30min"] + ["distribution_a_1h", "polarity_a_1h", "sentiment_a_1h"] + ["distribution_a_6h", "polarity_a_6h", "sentiment_a_6h"]
    # db_averages
    labels += ["distribution_t_15min", "polarity_t_15min", "sentiment_t_15min"] + ["distribution_c_15min", "polarity_c_15min", "sentiment_c_15min"]
    labels += ["distribution_t_30min", "polarity_t_30min", "sentiment_t_30min"] + ["distribution_c_30min", "polarity_c_30min", "sentiment_c_30min"]
    labels += ["distribution_t_1h", "polarity_t_1h", "sentiment_t_1h"] + ["distribution_c_1h", "polarity_c_1h", "sentiment_c_1h"]
    labels += ["distribution_t_6h", "polarity_t_6h", "sentiment_t_6h"] + ["distribution_c_6h", "polarity_c_6h", "sentiment_c_6h"]
    # technical data
    labels += ["price_15min", "volume_15min", "price_all_15min", "price_30min", "volume_30min", "price_all_30min", "price_1h", "volume_1h", "price_all_1h", "price_6h", "volume_6h", "volume_all_6h"]
    # topics
    labels += ["topic_" + str(i) for i in range(len(articles[0]["topics"]))]
    labels += ["average_topic_" + str(i) for i in range(len(articles[0]["topics"]))]
    # tfidf
    labels += vocabulary

    return X, Y, IDs, labels


def create_X(client, i, article_data, weights):
    date_from = int(time.mktime(article_data["date"].timetuple()) + 3600)

    technical_data = []
    db_averages = []
    data_averages = []
    average_tfidf, n, average_topics = sparse.csr_matrix([]), 0, np.array([])

    time_windows = [900, 1800, 2*3600, 6*3600]                                          # 15 min, 60 min, 6h
    the_window = 1800
    for time_window in time_windows:
        technical_data.append(common.get_price_change(client, article_data["currency"], date_from - time_window, date_from))
        technical_data.append(common.get_total_volume(client, article_data["currency"], date_from - time_window, date_from) / common.get_total_volume(client, "all", date_from - time_window, date_from))
        technical_data.append(common.get_all_price_changes(client, date_from - time_window, date_from))

        db_averages += common.get_averages_from_db(client, article_data["date"], time_window, article_data["currency"], articles=False)
        if time_window != the_window:
            data_averages += common.get_averages_from_data(articles, article_data["date"], time_window, article_data["currency"], i, threshold=0.0, type="article", data_averages_only=True)
        else:
            _data_averages, average_tfidf, n, average_topics = common.get_averages_from_data(articles, article_data["date"], time_window, article_data["currency"], i, threshold=0.0, type="article", data_averages_only=False)
            data_averages += _data_averages

    _X = [1 / (n + 1), article_data["title_sentiment"], article_data["reduced_text_sentiment"], article_data["title_polarity"], article_data["reduced_text_polarity"], article_data["currency_in_title"]] + data_averages + db_averages + technical_data

    if not np.all(np.isfinite(_X)):
        return None

    # topics = article_data["topics"]     # * (2 / (n + 1)) + average_topics * (n / (n + 1))
    tfidf = article_data["tfidf"] * (1 / (n + 1)) + (average_tfidf * (n / (n + 1))).multiply(article_data["tfidf"].power(0))
    tfidf = tfidf.multiply(weights)

    _X += list(article_data["topics"]) + list(average_topics) + tfidf.todense().tolist()[0]

    return sparse.csr_matrix(_X)


def create_Y(client, article_data, window, margin):
    date_from = int(time.mktime(article_data["date"].timetuple()) + 3600)
    date_to = date_from + window
    percent_change = common.get_min_max_price_change(client, article_data["currency"].lower(), date_from, date_to)

    if not np.isnan(percent_change):
        _Y = 0
        if percent_change > margin:
            _Y = 1
        elif percent_change <= -margin:
            _Y = -1
    else:
        return None

    return _Y


def get_Y(IDs, window, margin):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    ids = set(IDs)
    Y = []

    for article in articles:
        if article["_id"] in ids:
            _Y = create_Y(client, article, window, margin)
            if _Y is not None:
                Y.append(_Y)
            else:
                print("recompute IDs!")
                return None

    return Y


def get_relative_Y_changes(IDs, window):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    ids = set(IDs)
    p = []
    currencies = []

    for article in articles:
        if article["_id"] in ids:
            print("a")
            date_from = int(time.mktime(article["date"].timetuple()) + 3600)
            date_to = date_from + window
            _p = common.get_min_max_price_change(client, article["currency"], date_from, date_to)
            if not np.isnan(_p):
                p.append(_p)
                currencies.append(article["currency"].lower())
            else:
                print("recompute IDs")
                print(article["_id"])
                return None, None

    return p, currencies
