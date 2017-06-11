from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from old.news.Article import Article


def load_articles(db_news, currency, date_from, date_to, p=False):
    articles = []
    currency = currency.upper()

    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    sentiment_analyzer = SentimentIntensityAnalyzer()

    for article in db_news.articles.find({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lte": date_to}}, {"$or": [{"relevant": True}, {"relevant_predict": True}]}, {"currency": currency}]}):
        if "stemmed_text" not in article or "stemmed_title" not in article:
            # stem text & title
            text = tokenizer.tokenize(article["text"])
            title = tokenizer.tokenize(article["title"])

            text = [stemmer.stem(word.lower()) for word in text if word not in stopwords.words("english")]
            title = [stemmer.stem(word.lower()) for word in title if word not in stopwords.words("english")]

            db_news.articles.update({"_id": article["_id"]}, {"$set": {"stemmed_text": text, "stemmed_title": title}})
            article["stemmed_title"] = title
            article["stemmed_text"] = text

        if "sentiment_text" not in article or "sentiment_title" not in article:
            # calc sentiment
            scores_text = sentiment_analyzer.polarity_scores(article["text"])
            scores_title = sentiment_analyzer.polarity_scores(article["title"])

            db_news.articles.update({"_id": article["_id"]}, {"$set": {"sentiment_text": scores_text, "sentiment_title": scores_title}})
            article["sentiment_text"] = scores_text
            article["sentiment_title"] = scores_title

        _article = create_article(article)
        articles.append(_article)

    if p:
        for article in articles:
            print(article)

    return articles


def load_articles_from_ids(db_news, article_ids):
    articles = []
    for key in article_ids:
        article = db_news.articles.find_one({"_id": key})
        _article = create_article(article)
        articles.append(_article)
    return articles


def create_article(article):
    _article = Article(article["_id"], currencies=[article["currency"].lower()], date=article["date"], title=article["title"], text=article["text"], authors=article["authors"], source=article["source"])
    _article.stemmed_text = article["stemmed_text"]
    _article.stemmed_title = article["stemmed_title"]
    _article.sentiment_title = article["sentiment_title"]
    _article.sentiment_text = article["sentiment_text"]
    _article.map_currencies()
    return _article
