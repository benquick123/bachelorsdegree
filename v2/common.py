from nltk.corpus import stopwords as sw
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import lda
import numpy as np
import re
from datetime import timedelta
import csv

financial_dict = dict()


def create_financial_dict():
    file = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/texts/financial_words.csv")
    read = csv.reader(file, delimiter=";")
    stemmer = PorterStemmer()
    pattern = re.compile("[\W_]+")
    next(read)
    for line in read:
        word = remove_special_chars(line[3], pattern)
        word = stemmer.stem(word)
        financial_dict[word] = {"ratio": float(line[7]), "disp": float(line[8])}


def create_financial_vocabulary(vocabulary):
    m = np.zeros(len(vocabulary))
    for key, index in vocabulary.items():
        if key in financial_dict:
            m[index] = financial_dict[key]["ratio"]
    return m


def stem(word_list):
    stemmer = PorterStemmer()

    _word_list = []
    for word in word_list:
        try:
            _word_list.append(stemmer.stem(word))
        except IndexError:
            continue

    return _word_list


def remove_special_chars(word, pattern):
    # @, #, ?, ,, ;, ., :, ...
    word = pattern.sub("", word)
    if len(word) == 0:
        return None
    return word


def get_pos_tags(tokens):
    pos_tags = [tag[1] for tag in pos_tag(tokens)]
    return pos_tags


def calc_sentiment_polarity(word_list, pos_tags):
    tags_dict = {"N": "n", "V": "v", "J": "a", "R": "r"}

    pos_scores = []
    neg_scores = []

    lemmatizer = WordNetLemmatizer()

    k = 0
    for i, word in enumerate(zip(word_list, pos_tags)):
        try:
            tag = tags_dict[word[1][0]]
            word = lemmatizer.lemmatize(word[0], pos=tag)
            query = ".".join([word, tag, "01"])
            try:
                s = swn.senti_synset(query)
                k += 1
            except:
                s = list(swn.senti_synsets(word))
                if len(s) > 0:
                    s = s[0]
                else:
                    continue

            if i == 0 or word_list[i-1] != "not":
                neg_scores.append(s.neg_score())
                pos_scores.append(s.pos_score())
            else:
                neg_scores.append(-s.neg_score())
                pos_scores.append(-s.pos_score())
        except KeyError:
            pass

    if k > 0:
        sentiment = (np.sum(pos_scores) - np.sum(neg_scores)) / k
        polarity = np.std(np.append(pos_scores, neg_scores*-1))
    else:
        sentiment = 0
        polarity = 0
    return sentiment, polarity


def is_stop_word(word):
    stopwords = set(sw.words("english"))
    if word in stopwords:
        return True
    else:
        return False


def calc_tf_idf(texts, dict_key, is_conversation=False):
    def analyzer(text):
        if is_conversation:
            _text = []
            for t in text["messages"]:
                _text += t[dict_key]
            return _text
        return text[dict_key]

    vectorizer = TfidfVectorizer(analyzer=analyzer)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer.vocabulary_


def generate_topics(texts, dict_key, n_topics, is_conversation=False):
    def analyzer(text):
        if is_conversation:
            _text = []
            for t in text["messages"]:
                _text += t[dict_key]
            return _text
        return text[dict_key]

    vectorizer = CountVectorizer(analyzer=analyzer)
    count_matrix = vectorizer.fit_transform(texts)

    model = lda.LDA(n_topics=n_topics)
    topics = model.fit_transform(count_matrix)

    return topics.tolist()


def get_price_change(client, currency, date_from, date_to):
    db = client["crypto_prices"]
    try:
        start_price = db[currency.lower()].find_one({"_id": date_from - (date_from % 300) + 300}, {"_id": 0, "weightedAverage": 1})["weightedAverage"]
        end_price = db[currency.lower()].find_one({"_id": {"$lte": date_to - (date_to % 300) + 300}}, {"_id": 0, "weightedAverage": 1})["weightedAverage"]
        percent_change = (end_price - start_price) / start_price
        return percent_change
    except TypeError:
        return None


def get_averaged_results(client, date_to, window, currency, articles=True, tweets=True, conversations=True):
    date_from = date_to - timedelta(seconds=window)

    averages = []
    if articles:
        sentiment = []
        polarity = []
        predictions = []
        db = client["news"]
        n = db.articles.count({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lt": date_to}}, {"currency": currency.upper()}, {"relevant": True}]})
        for article in db.articles.find({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lt": date_to}}, {"currency": currency.upper()}, {"relevant": True}]}, {"_id": 0, "text_sentiment": 1, "text_polarity": 1, "prediction": 1}):
            sentiment.append(article["text_sentiment"])
            polarity.append(article["text_polarity"])
            if "prediction" in article:
                predictions.append(article["prediction"])

        averages.append(n)
        averages.append(np.mean(polarity) if len(polarity) > 0 else 0)
        averages.append(np.mean(sentiment) if len(sentiment) > 0 else 0)
        averages.append(np.mean(predictions) if len(predictions) > 0 else 0)
    if tweets:
        sentiment = []
        polarity = []
        predictions = []
        db = client["twitter"]
        n = db.messages.count({"$and": [{"posted_time": {"$gte": date_from}}, {"posted_time": {"$lt": date_to}}, {"crypto_currency": currency.upper()}, {"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}]})
        for tweet in db.messages.find({"$and": [{"posted_time": {"$gte": date_from}}, {"posted_time": {"$lt": date_to}}, {"crypto_currency": currency.upper()}, {"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}]}, {"_id": 0, "sentiment": 1, "polarity": 1, "prediction": 1}):
            sentiment.append(tweet["sentiment"])
            polarity.append(tweet["polarity"])
            if "prediction" in tweet:
                predictions.append(tweet["prediction"])

        averages.append(n)
        averages.append(np.mean(polarity) if len(polarity) > 0 else 0)
        averages.append(np.mean(sentiment) if len(sentiment) > 0 else 0)
        averages.append(np.mean(predictions) if len(predictions) > 0 else 0)
    if conversations:
        sentiment = []
        polarity = []
        predictions = []
        db = client["trollbox"]
        date_to = date_to.strftime("%Y-%m-%d %H:%M:%S UTC")
        date_from = date_from.strftime("%Y-%m-%d %H:%M:%S UTC")
        n = db.conversations.count({"$and": [{"conversation_start": {"$gte": date_from}}, {"conversation_end": {"$lt": date_to}}, {"coin_mentions": currency.lower()}]})
        for conversation in db.conversations.find({"$and": [{"conversation_start": {"$gte": date_from}}, {"conversation_end": {"$lt": date_to}}, {"coin_mentions": currency.lower()}]}, {"_id": 0, "avg_sentiment": 1, "avg_polarity": 1, "prediction": 1}):
            sentiment.append(conversation["avg_sentiment"])
            polarity.append(conversation["avg_polarity"])
            if "prediction" in conversation:
                predictions.append(conversation["prediction"])

        averages.append(n)
        averages.append(np.mean(polarity) if len(polarity) > 0 else 0)
        averages.append(np.mean(sentiment) if len(sentiment) > 0 else 0)
        averages.append(np.mean(predictions) if len(predictions) > 0 else 0)
    return averages

create_financial_dict()
