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
from scipy import sparse

financial_dict = dict()
currencies = dict()


def create_currencies_dict():
    file = open("/home/ubuntu/diploma/Proletarian 1.0/v2/other/currencies.csv")
    read = csv.reader(file, delimiter=";")

    for line in read:
        currencies[line[0].lower()] = line[1].lower()


def create_financial_dict():
    min_ratio = float("inf")
    max_ratio = float("-inf")

    file = open("/home/ubuntu/diploma/Proletarian 1.0/v2/texts/financial_words.csv")
    read = csv.reader(file, delimiter=";")
    # stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    pattern = re.compile("[\W_]+")
    next(read)
    for line in read:
        word = remove_special_chars(line[3], pattern)
        word = lemmatizer.lemmatize(word)
        ratio = float(line[7])
        disp = float(line[8])

        financial_dict[word] = {"ratio": ratio, "disp": disp}

        if min_ratio > ratio:
            min_ratio = ratio
        if max_ratio < ratio:
            max_ratio = ratio

    to_delete = None
    for key in financial_dict.keys():
        financial_dict[key]["ratio"] = (financial_dict[key]["ratio"] - min_ratio) / (max_ratio - min_ratio)
        if financial_dict[key]["ratio"] == 1.0:
            to_delete = key

    if to_delete is not None:
        del financial_dict[key]


def create_financial_vocabulary(vocabulary):
    m = []
    for word in vocabulary:
        word = word.split(".")[0]
        if word in financial_dict:
            m.append(financial_dict[word]["ratio"])
        else:
            m.append(0)
    return np.array(m)


def stem(word_list):
    stemmer = PorterStemmer()

    _word_list = []
    for word in word_list:
        try:
            _word_list.append(stemmer.stem(word))
        except IndexError:
            continue

    return _word_list


def lemmatize(word_list, pos_tags):
    tags_dict = {"N": "n", "V": "v", "J": "a", "R": "r"}
    lemmatizer = WordNetLemmatizer()

    _word_list = []

    for word, pos_tag in zip(word_list, pos_tags):
        try:
            tag = tags_dict[pos_tag[0]]
            _word_list.append(lemmatizer.lemmatize(word, pos=tag) + "." + tag)
        except KeyError:
            continue

    return _word_list


def remove_special_chars(word, pattern):
    # @, #, ?, ,, ;, ., :, ...
    word = pattern.sub("", word)
    if len(word) == 0:
        return None
    return word


def get_pos_tags(tokens):
    pos_tags = pos_tag(tokens)
    pos_tags = [tag for word, tag in pos_tags]
    return pos_tags


def calc_sentiment_polarity(word_list):
    # tags_dict = {"N": "n", "V": "v", "J": "a", "R": "r"}
    offsets = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    tags = ["n", "v", "a", "r"]

    pos_scores = []
    neg_scores = []

    # lemmatizer = WordNetLemmatizer()

    for i, word in enumerate(word_list):
        try:
            word, tag = tuple(word.split("."))
            s = None
            for offset in offsets:
                query = word + "." + tag + "." + offset
                try:
                    s = swn.senti_synset(query)
                    break
                except:
                    continue

            if s is None:
                _tags = list(tags)
                _tags.remove(tag)
                for tag in _tags:
                    for offset in offsets:
                        query = word + "." + tag + "." + offset
                        try:
                            s = swn.senti_synset(query)
                            break
                        except:
                            continue

                    if s is not None:
                        break

            if s is None:
                neg_scores.append(0)
                pos_scores.append(0)
            elif i == 0 or word_list[i-1] != "not":
                neg_scores.append(s.neg_score())
                pos_scores.append(s.pos_score())
            else:
                neg_scores.append(-s.neg_score())
                pos_scores.append(-s.pos_score())
        except KeyError:
            pass

    if len(pos_scores) > 0:
        sentiment = (np.sum(pos_scores) - np.sum(neg_scores)) / len(pos_scores)
        polarity = np.std(np.append(pos_scores, neg_scores*-1))
    else:
        sentiment = 0
        polarity = 0
    return sentiment, polarity


def calc_sentiment_for_vocabulary(vocabulary):
    sentiments = []
    offsets = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    tags = ["n", "v", "a", "r"]

    for word in vocabulary:
        word, tag = tuple(word.split("."))
        s = None
        for offset in offsets:
            query = word + "." + tag + "." + offset
            try:
                s = swn.senti_synset(query)
                break
            except:
                continue

        if s is None:
            _tags = list(tags)
            _tags.remove(tag)
            for tag in _tags:
                for offset in offsets:
                    query = word + "." + tag + "." + offset
                    try:
                        s = swn.senti_synset(query)
                        break
                    except:
                        continue

                if s is not None:
                    break

        if s is not None:
            sentiments.append(s.pos_score() - s.neg_score())
        else:
            sentiments.append(0)

    return np.array(sentiments)


def is_stop_word(word):
    stopwords = set(sw.words("english"))
    if word in stopwords:
        return True
    else:
        return False


def calc_tf_idf(texts, min_df, max_df, dict_key, is_conversation=False):
    def analyzer(text):
        if is_conversation:
            _text = []
            for t in text["messages"]:
                _text += t[dict_key]
            return _text
        return text[dict_key]

    vectorizer = TfidfVectorizer(analyzer=analyzer, min_df=min_df, max_df=max_df)
    tfidf_matrix = vectorizer.fit_transform(texts)
    """from operator import itemgetter
    print(sorted(zip(vectorizer.get_feature_names(), vectorizer.idf_), key=itemgetter(1)))
    print(len(vectorizer.get_feature_names()))
    exit()"""
    return tfidf_matrix, vectorizer.get_feature_names()


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

    return topics


def get_total_volume(client, currency, date_from, date_to):
    db = client["crypto_prices"]
    date_from = date_from - (date_from % 300) + 300
    date_to = date_to - (date_to % 300) + 300
    volume = 0
    for price_info in db[currency.lower()].find({"$and": [{"_id": {"$gte": date_from}}, {"_id": {"$lte": date_to}}]}, {"_id": 0, "volume": 1}):
        volume += price_info["volume"]

    return volume


def get_all_price_changes(client, date_from, date_to):
    db = client["crypto_prices"]
    date_from = date_from - (date_from % 300) + 300
    date_to = date_to - (date_to % 300) + 300
    price = 1
    for price_info in db["all"].find({"$and": [{"_id": {"$gt": date_from}}, {"_id": {"$lte": date_to}}]}, {"_id": 0, "weightedAverage": 1}):
        price *= price_info["weightedAverage"]

    return price - 1


def get_price_change(client, currency, date_from, date_to):
    db = client["crypto_prices"]
    date_from = date_from - (date_from % 300) + 300
    date_to = date_to - (date_to % 300) + 300
    try:
        start_price = db[currency.lower()].find_one({"_id": date_from}, {"_id": 0, "weightedAverage": 1})["weightedAverage"]
        end_price = db[currency.lower()].find_one({"_id": date_to}, {"_id": 0, "weightedAverage": 1})["weightedAverage"]

        percent_change = (end_price - start_price) / start_price
        return percent_change
    except TypeError:
        return np.nan


def get_min_max_price_change(client, currency, date_from, date_to):
    db = client["crypto_prices"]
    date_from = date_from - (date_from % 300) + 300
    date_to = date_to - (date_to % 300) + 300
    try:
        start_price = db[currency.lower()].find_one({"_id": date_from}, {"_id": 0, "open": 1})["open"]
        min_price = list(db[currency.lower()].find({"$and": [{"_id": {"$gte": date_from}}, {"_id": {"$lte": date_to}}]}, {"_id": 0, "close": 1}).sort("close", 1).limit(1))[0]["close"]
        max_price = list(db[currency.lower()].find({"$and": [{"_id": {"$gte": date_from}}, {"_id": {"$lte": date_to}}]}, {"_id": 0, "close": 1}).sort("close", -1).limit(1))[0]["close"]

        min_percent_change = (min_price - start_price) / start_price
        max_percent_change = (max_price - start_price) / start_price
        if abs(min_percent_change) > abs(max_percent_change):
            return min_percent_change
        else:
            return max_percent_change
    except TypeError or IndexError:
        return np.nan


def get_averages_from_data(data, date_to, window, currency, k, threshold, type, data_averages_only=False):
    currency_key = ""
    date_key = ""
    sentiment_key = ""
    polarity_key = ""
    if type == "article":
        currency_key = "currency"
        date_key = "date"
        sentiment_key = "reduced_text_sentiment"
        polarity_key = "reduced_text_polarity"
    elif type == "tweet":
        currency_key = "crypto_currency"
        date_key = "posted_time"
        sentiment_key = "sentiment"
        polarity_key = "polarity"
    elif type == "conversation":
        currency_key = "coin_mentions"
        date_key = "conversation_end"
        sentiment_key = "avg_sentiment"
        polarity_key = "avg_polarity"

    date_from = date_to - timedelta(seconds=window)

    topics = []
    tfidf = None
    weights = []
    sentiment = []
    polarity = []
    n = 0

    for i in range(k-1, -1, -1):
        if data[i][date_key] >= date_from:
            if (type != "conversation" and data[i][currency_key].lower() == currency.lower()) or (currency.lower() in set(data[i][currency_key])):
                if not data_averages_only:
                    if "topics" in data[i]:
                        topics.append(data[i]["topics"])

                    if tfidf is None and "tfidf" in data[i]:
                        tfidf = sparse.csr_matrix(data[i]["tfidf"])
                    elif "tfidf" in data[i]:
                        tfidf = sparse.vstack([tfidf, data[i]["tfidf"]])
                    else:
                        print("TF-IDF not in data[" + str(i) + "]")

                sentiment.append(data[i][sentiment_key] if np.isfinite(data[i][sentiment_key]) else 0)
                polarity.append(data[i][polarity_key] if np.isfinite(data[i][polarity_key]) else 0)
                weights.append((data[i][date_key] - date_from).total_seconds() / window)
                n += 1
        else:
            break

    if not data_averages_only:
        if tfidf is None or tfidf.shape[0] == 0:
            tfidf = sparse.csr_matrix((1, data[k]["tfidf"].shape[1] if "tfidf" in data[k] else 5))
        else:
            _weights = sparse.lil_matrix((len(weights), len(weights)))
            _weights.setdiag(weights)
            tfidf = _weights * tfidf
            tfidf = tfidf.mean(axis=0)
            tfidf = sparse.csr_matrix(np.where(tfidf > threshold, tfidf, 0)[0])

    sentiment = np.average(sentiment, weights=weights) if len(sentiment) > 0 and sum(weights) > 0 else 0
    polarity = np.average(polarity, weights=weights) if len(polarity) > 0 and sum(weights) > 0 else 0
    distribution = np.average(weights) if len(weights) > 0 else 0

    if data_averages_only:
        return [distribution, polarity, sentiment]

    if "topics" in data[0]:
        if len(topics) == 0 or sum(weights) == 0:
            topics = np.zeros(data[k]["topics"].shape[0])
        else:
            topics = np.average(topics, axis=0, weights=weights)
        return [distribution, polarity, sentiment], tfidf, n, np.array(topics)
    else:
        return [distribution, polarity, sentiment], tfidf, n, None


def get_averages_from_db(client, date_to, window, currency, articles=True, tweets=True, conversations=True):
    date_from = date_to - timedelta(seconds=window)

    averages = []
    if articles:
        sentiment = []
        polarity = []
        weights = []
        db = client["news"]

        for article in db.articles.find({"$and": [{"date": {"$gte": date_from}}, {"date": {"$lt": date_to}}, {"currency": currency.upper()}, {"relevant": True}]}, {"_id": 0, "text_sentiment": 1, "text_polarity": 1, "date": 1}):
            if article["date"].hour != 0 or article["date"].minute != 0 or article["date"].second != 0:
                sentiment.append(article["text_sentiment"] if np.isfinite(article["text_sentiment"]) else 0)
                polarity.append(article["text_polarity"] if np.isfinite(article["text_polarity"]) else 0)
                weights.append((article["date"] - date_from).total_seconds() / window)

        distribution = np.average(weights) if len(weights) > 0 else 0

        averages.append(distribution)
        averages.append(np.average(polarity, weights=weights) if len(polarity) > 0 and sum(weights) > 0 else 0)
        averages.append(np.average(sentiment, weights=weights) if len(sentiment) > 0 and sum(weights) > 0 else 0)
    if tweets:
        sentiment = []
        polarity = []
        weights = []
        db = client["twitter"]

        for tweet in db.messages.find({"$and": [{"posted_time": {"$gte": date_from}}, {"posted_time": {"$lt": date_to}}, {"crypto_currency": currency.upper()}, {"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}]}, {"_id": 0, "sentiment": 1, "polarity": 1, "posted_time": 1}):
            sentiment.append(tweet["sentiment"] if np.isfinite(tweet["sentiment"]) else 0)
            polarity.append(tweet["polarity"] if np.isfinite(tweet["polarity"]) else 0)
            weights.append((tweet["posted_time"] - date_from).total_seconds() / window)

        distribution = np.average(weights) if len(weights) > 0 else 0

        averages.append(distribution)
        averages.append(np.average(polarity, weights=weights) if len(polarity) > 0 and sum(weights) > 0 else 0)
        averages.append(np.average(sentiment, weights=weights) if len(sentiment) > 0 and sum(weights) > 0 else 0)
    if conversations:
        sentiment = []
        polarity = []
        weights = []
        db = client["trollbox"]

        for conversation in db.conversations.find({"$and": [{"conversation_start": {"$gte": date_from}}, {"conversation_end": {"$lt": date_to}}], "coin_mentions": currency.lower(), "messages_len": {"$gte": 1}}, {"_id": 0, "avg_sentiment": 1, "avg_polarity": 1, "conversation_start": 1}):
            sentiment.append(conversation["avg_sentiment"] if np.isfinite(conversation["avg_sentiment"]) else 0)
            polarity.append(conversation["avg_polarity"] if np.isfinite(conversation["avg_polarity"]) else 0)
            weights.append((conversation["conversation_start"] - date_from).total_seconds() / window)

        distribution = np.average(weights) if len(weights) > 0 else 0

        averages.append(distribution)
        averages.append(np.average(polarity, weights=weights) if len(polarity) > 0 and sum(weights) > 0 else 0)
        averages.append(np.average(sentiment, weights=weights) if len(sentiment) > 0 and sum(weights) > 0 else 0)

    return averages

create_financial_dict()
create_currencies_dict()
