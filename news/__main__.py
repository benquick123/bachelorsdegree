import pymongo
from news.Article import Article
from datetime import datetime
import pytz


def load_articles(db_news, currency, date_from, date_to):
    articles = []

    for article in db_news.articles.find({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lte": date_to}}]}).sort("_id", 1):
        if (article["_id"].split(":")[0] == currency or currency == "*") and len(article["title"]) > 0 and len(article["text"]) > 0:
            date = article["date"]
            title = article["title"]
            text = article["text"]
            authors = article["authors"]
            source = article["source"]
            a = Article(date, title, text, authors, source)

            a.stemmed_text = article["stemmed_text"]
            a.stemmed_title = article["stemmed_title"]

            articles.append(a)

    return articles


def __main__():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db_news = client["news"]
    currency = "ETH"

    date_from = datetime(2015, 11, 1, 0, 0, 0).replace(tzinfo=pytz.utc)
    date_to = datetime(2016, 10, 31, 23, 59, 59).replace(tzinfo=pytz.utc)

    articles = load_articles(db_news, currency, date_from, date_to)

__main__()
