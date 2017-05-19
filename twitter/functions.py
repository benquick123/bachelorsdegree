from twitter.Tweet import Tweet

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_tweets(db_twitter, currency, date_from, date_to, p=False, _tweets=None):
    tweets = []
    currency = currency.upper()

    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    sentiment_analyzer = SentimentIntensityAnalyzer()

    for tweet in (db_twitter.messages.find({"$and": [{"posted_time": {"$gte": date_from}}, {"posted_time": {"$lte": date_to}}, {"crypto_currency": currency}]}) if _tweets is None else _tweets):
        if _tweets is None or (date_from <= tweet["posted_time"] <= date_to):
            try:
                if "stemmed_text" not in tweet:
                    # stem text
                    text = tokenizer.tokenize(tweet["text"])
                    text = [stemmer.stem(word.lower()) for word in text if word not in stopwords.words("english")]

                    db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"stemmed_text": text}})
                    tweet["stemmed_text"] = text

                if "sentiment_text" not in tweet:
                    # calc sentiment
                    scores_text = sentiment_analyzer.polarity_scores(tweet["text"])

                    db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"sentiment_text": scores_text}})
                    tweet["sentiment_text"] = scores_text

                _tweet = create_tweet(tweet)
                tweets.append(_tweet)
            except IndexError:
                for i in range(100):
                    print("NOT ENOUGH WORDS. SKIPPED.")

    if p:
        for tweet in tweets:
            print(tweet)

    return tweets


def load_tweets_from_ids(db_twitter, tweet_ids):
    tweets = []
    for key in tweet_ids:
        tweet = db_twitter.messages.find_one({"_id": key})
        _tweet = create_tweet(tweet)
        tweets.append(_tweet)
    return tweets


def create_tweet(tweet):
    _tweet = Tweet(tweet_id=tweet["_id"], currencies=[tweet["crypto_currency"].lower()], date=tweet["posted_time"], text=tweet["text"], author_id=tweet["author"])
    _tweet.stemmed_text = tweet["stemmed_text"]
    _tweet.sentiment_text = tweet["sentiment_text"]
    _tweet.map_currencies()

    return _tweet
