import pymongo
import v2.common as common
import re


def load_with_attr():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["news"]

    articles = []
    for article in db.articles.find({"relevant": True}, {"_id": 1, "title": 1, "date": 1, "source": 1, "authors": 1, "currency": 1}):
        if len(article["clean_text"]) > 1 and len(article["clean_title"]) > 1:
            articles.append(article)

    return articles


def load_without_attr(save_to_db=True):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["news"]

    pattern = re.compile("[\W_]+")

    articles = []
    for article in db.articles.find({"relevant": True}, {"_id": 1, "title": 1, "date": 1, "source": 1, "authors": 1, "currency": 1, "clean_title": 1, "clean_text": 1, "title_sentiment": 1, "title_polarity": 1, "text_sentiment": 1, "text_polarity": 1, "topic": 1}):
        title = []
        text = []

        for word in article["title"].split(" "):
            word = word.lower()

            word = common.remove_special_chars(word, pattern)
            if word is None:
                continue

            if not common.is_stop_word(word):
                title.append(word)

        for word in article["text"].split(" "):
            word = word.lower()

            word = common.remove_special_chars(word, pattern)
            if word is None:
                continue

            if not common.is_stop_word(word):
                text.append(word)

        title_sentiment, title_polarity = common.calc_sentiment_polarity(title)
        text_sentiment, text_polartiy = common.calc_sentiment_polarity(text)

        title = common.stem(title)
        text = common.stem(text)

        topic = extract_topic(text)

        article["clean_title"] = title
        article["clean_text"] = text
        article["title_sentiment"] = title_sentiment
        article["text_sentiment"] = text_sentiment
        article["title_polartiy"] = title_polarity
        article["text_polartiy"] = text_polartiy
        article["topic"] = topic

        if save_to_db:
            db.articles.update_one({"_id": article["_id"]}, {"$set": {"clean_title": title, "clean_text": text, "title_sentiment": title_sentiment, "title_polarity": title_polarity, "text_sentiment": text_sentiment, "text_polarity": text_polartiy, "topic": topic}})

        if len(title) > 1 and len(text) > 1:
            articles.append(article)

    return articles


def extract_topic(word_list):
    return []


def create_matrix(articles):
    X = []
    Y = []
    for article in articles:
        _X, _Y = create_matrix_line(article)
        X.append(_X)
        Y.append(_Y)


def create_matrix_line(article_data):
    _X = []
    _Y = 0
    return _X, _Y

