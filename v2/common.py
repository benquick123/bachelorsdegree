import pymongo
import re
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer


def stem(word_list):
    stemmer = PorterStemmer()

    _word_list = []
    for word in word_list:
        _word_list.append(stemmer.stem(word))

    return _word_list


def remove_special_chars(word, pattern):
    # @, #, ?, ,, ;, ., :, ...
    word = pattern.sub("", word)
    if len(word) == 0:
        return None
    return word


def calc_sentiment_polarity(word_list):
    return 0, 0


def is_stop_word(word):
    stopwords = set(sw.words("english"))
    if word in stopwords:
        return True
    else:
        return False


def calc_tf_idf(texts):
    pass


def get_averaged_results(window, articles=True, tweets=True, conversations=True):
    averages = []
    if articles:
        averages.append(get_average_freq(window, "articles"))
        averages.append(get_average_polarity(window, "articles"))
        averages.append(get_average_sentiment(window, "articles"))
        averages.append(get_average_predictions(window, "articles"))
    if tweets:
        averages.append(get_average_freq(window, "tweets"))
        averages.append(get_average_polarity(window, "tweets"))
        averages.append(get_average_sentiment(window, "tweets"))
        averages.append(get_average_predictions(window, "tweets"))
    if conversations:
        averages.append(get_average_freq(window, "conversations"))
        averages.append(get_average_polarity(window, "conversations"))
        averages.append(get_average_sentiment(window, "conversations"))
        averages.append(get_average_predictions(window, "conversations"))

    return averages


def get_average_freq(window, text_type):
    if text_type == "articles":
        pass
    elif text_type == "tweets":
        pass
    elif text_type == "conversations":
        pass
    return None


def get_average_sentiment(window, text_type):
    if text_type == "articles":
        pass
    elif text_type == "tweets":
        pass
    elif text_type == "conversations":
        pass
    return None


def get_average_polarity(window, text_type):
    if text_type == "articles":
        pass
    elif text_type == "tweets":
        pass
    elif text_type == "conversations":
        pass
    return None


def get_average_predictions(window, text_type):
    if text_type == "articles":
        pass
    elif text_type == "tweets":
        pass
    elif text_type == "conversations":
        pass
    return None
