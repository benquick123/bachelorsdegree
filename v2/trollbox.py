import pymongo
import v2.common as common
import re
import numpy as np
from datetime import datetime
import time
from scipy import sparse

conversations = []


def load_with_attr(n=None, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["trollbox"]

    i = 0
    for conversation in db.conversations.find({"coin_mentions.0": {"$exists": True}}, {"_id": 1, "conversation_end": 1, "coin_mentions": 1, "messages_len": 1, "avg_sentiment": 1, "avg_polarity": 1}, no_cursor_timeout=True).sort("conversation_end", 1):
        if n is None or i < n:
            if p:
                print("processing conversation", i, "(" + str(conversation["_id"]) + ")")

            conversation_tree = build_tree(conversation["_id"], db)
            messages = get_messages(conversation_tree)
            if len(messages) > 0:
                conversation["messages"] = messages
                conversations.append(conversation)
            i += 1
        else:
            break

    return conversations


def load_without_attr(p=False, save_to_db=True):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["trollbox"]

    pattern = re.compile("[\W]+")

    # conversations = []
    i = 0
    for conversation in db.conversations.find({"coin_mentions.0": {"$exists": True}}, {"_id": 1, "conversation_end": 1, "coin_mentions": 1}, no_cursor_timeout=True).sort("conversation_end", 1):
        if p:
            print("processing conversation", i, "(" + str(conversation["_id"]) + ")")
            i += 1

        conversation_tree = build_tree(conversation["_id"], db)

        messages = []
        avg_sentiment = []
        avg_polarity = []
        for message in get_messages(conversation_tree):
            message_text = message["text"].split(" ")
            word = remove_username(message_text[0])
            if word is None:
                del message_text[0]

            text = []
            for word in message_text:
                word = word.lower()
                word = common.remove_special_chars(word, pattern)
                if word is None:
                    continue

                # if not common.is_stop_word(word) or word == "not":
                text.append(word)

            pos_tag_text = common.get_pos_tags(text)
            text = common.lemmatize(text, pos_tag_text)
            sentiment, polarity = common.calc_sentiment_polarity(text)

            message["clean_text"] = text
            message["pos_tag_text"] = pos_tag_text
            message["sentiment"] = sentiment
            message["polarity"] = polarity

            # TODO: weight polarity and sentiment by user reputation
            avg_sentiment.append(sentiment if np.isfinite(sentiment) else 0)
            avg_polarity.append(polarity if np.isfinite(polarity) else 0)

            if save_to_db:
                db.messages.update_one({"_id": message["_id"]}, {"$set": {"clean_text": text, "sentiment": sentiment, "polarity": polarity, "pos_tag_text": pos_tag_text}})

            if len(text) > 1:
                messages.append(message)

        if len(messages) > 0:
            avg_sentiment = np.mean(avg_sentiment) if len(avg_sentiment) > 0 else 0
            avg_polarity = np.mean(avg_polarity) if len(avg_polarity) > 0 else 0

            conversation["messages"] = messages
            conversation["messages_len"] = len(messages)
            conversation["avg_sentiment"] = avg_sentiment
            conversation["avg_polarity"] = avg_polarity

            if save_to_db:
                db.conversations.update_one({"_id": conversation["_id"]}, {"$set": {"messages_len": len(messages), "avg_polarity": avg_polarity, "avg_sentiment": avg_sentiment}})

            # conversations.append(conversation)

    return None     # conversations


def build_tree(_id, db):
    tree = db.messages.find_one({"_id": _id}, {"_id": 1, "user_id": 1, "date": 1, "text": 1, "responses": 1, "sentiment": 1, "polarity": 1, "clean_text": 1, "pos_tag_text": 1}, no_cursor_timeout=True)

    responses = tree["responses"]
    tree["responses"] = []
    for response_id in responses:
        to_append = build_tree(response_id, db)
        tree["responses"].append(to_append)

    return tree


def get_messages(tree):
    messages = []
    responses = tree["responses"]
    del tree["responses"]

    if "clean_text" not in tree or ("clean_text" in tree and len(tree["clean_text"]) > 1):
        messages.append(tree)

    for response in responses:
        messages += get_messages(response)
    return messages


def remove_username(word):
    if len(word) > 0 and word[-1] == ",":
        return None
    return word


def get_user_reputation(client, message):
    db = client["trollbox"]
    reputation = db.users.find_one({"_id": message["user_id"]}, {"_id": 0, "reputation": 1})["reputation"]
    return int(reputation)


def create_matrix(conversations, window, margin, p=False):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    X = None
    Y = []

    tfidf, vocabulary = common.calc_tf_idf(conversations, min_df=0.000005, max_df=1.0, dict_key="clean_text", is_conversation=True)
    financial_weights = common.create_financial_vocabulary(vocabulary) + 1
    sentiment_weights = common.calc_sentiment_for_vocabulary(vocabulary)
    sentiment_weights = np.where(sentiment_weights >= 0, sentiment_weights + 1, sentiment_weights - 1)
    weights = sparse.csr_matrix(financial_weights * sentiment_weights)

    date_start = datetime(2015, 11, 2, 0, 0, 0)

    for i, (conversation, conversation_tfidf) in enumerate(zip(conversations, tfidf)):
        for currency in conversation["coin_mentions"]:
            if p:
                print("processing conversation", str(conversation["_id"]), "(" + currency + ")", i)

            conversation["tfidf"] = conversation_tfidf
            if conversation["conversation_end"] >= date_start:
                _X, _Y = create_matrix_line(client, i, conversation, weights, currency, window, margin)
                if _Y is not None and _X is not None:
                    if X is None:
                        X = sparse.csr_matrix(_X)
                    else:
                        X = sparse.vstack([X, _X])
                    Y.append(_Y)

    # X: text_weight, average_sentiment, average_polarity, avg_reputation, conversation_length, past_price_change, past_distribution, past_polarity, past_sentiment,
    # past_article_distribution, past_article_polarity, past_article_sentiment, past_tweet_distribution, past_tweet_polarity, past_tweet_sentiment, weighted_tfidf
    return X, Y


def create_matrix_line(client, i, conversation_data, weights, currency, window, margin):
    date_from = int(time.mktime(conversation_data["conversation_end"].timetuple()) + 3600)
    date_to = date_from + window
    percent_change = common.get_min_max_price_change(client, currency, date_from, date_to)
    if percent_change is not None:
        _Y = 0
        if percent_change > margin:
            _Y = 1
        elif percent_change <= -margin:
            _Y = -1
    else:
        return None, None

    _price_change = common.get_price_change(client, currency, date_from-600, date_from)

    avg_reputation = []
    for message in conversation_data["messages"]:
        avg_reputation.append(get_user_reputation(client, message))

    db_averages = common.get_averages_from_db(client, conversation_data["conversation_end"], 600, currency, conversations=False)
    data_averages, average_tfidf, n, _ = common.get_averages_from_data(conversations, conversation_data["conversation_end"], 600, currency, i, 0, type="conversation")

    _X = [1 / (n+1), conversation_data["avg_sentiment"], conversation_data["avg_polarity"], np.mean(avg_reputation), conversation_data["messages_len"], _price_change] + data_averages + db_averages     # avg_reputation and messages_len are not in [-1, 1]

    if not np.all(np.isfinite(_X)):
        return None, None

    # tfidf = conversation_data["tfidf"]
    tfidf = conversation_data["tfidf"] * (2 / (n+1)) + (average_tfidf * (n / (n+1))).multiply(conversation_data["tfidf"].power(0))
    tfidf = tfidf.multiply(weights)
    _X += tfidf.todense().tolist()[0]

    return sparse.csr_matrix(_X), _Y
