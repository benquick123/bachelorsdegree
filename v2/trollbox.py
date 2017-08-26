import pymongo
import common as common
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
    IDs = []

    tfidf, vocabulary = common.calc_tf_idf(conversations, min_df=0.0, max_df=1.0, dict_key="clean_text", is_conversation=True)
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
                _Y = create_Y(client, conversation, currency, window, margin)
                if _Y is None:
                    continue

                _X = create_X(client, i, conversation, weights, currency)
                if _X is not None:
                    if X is None:
                        X = sparse.csr_matrix(_X)
                    else:
                        X = sparse.vstack([X, _X])
                    Y.append(_Y)
                    IDs.append(str(conversation["_id"]) + ":" + currency)

    # general attr
    labels = ["w", "avg_sentiment", "avg_polarity", "avg_reputation", "messages_len"]
    # averages
    labels += ["distribution_c_15min", "polarity_c_15min", "sentiment_c_15min"] + ["distribution_a_15min", "polarity_a_15min", "sentiment_a_15min"] + ["distribution_t_15min", "polarity_t_15min", "sentiment_t_15min"] + ["price_15min", "volume_15min", "price_all_15min"]
    labels += ["distribution_c_1h", "polarity_c_1h", "sentiment_c_1h"] + ["distribution_a_1h", "polarity_a_1h", "sentiment_a_1h"] + ["distribution_t_1h", "polarity_t_1h", "sentiment_t_1h"] + ["price_1h", "volume_1h", "price_all_1h"]
    labels += ["distribution_c_6h", "polarity_c_6h", "sentiment_c_6h"] + ["distribution_a_6h", "polarity_a_6h", "sentiment_a_6h"] + ["distribution_t_6h", "polarity_t_6h", "sentiment_t_6h"] + ["price_6h", "volume_6h", "volume_all_6h"]
    # tfidf
    labels += vocabulary

    return X, Y, IDs, labels


def create_X(client, i, conversation_data, weights, currency):
    date_from = int(time.mktime(conversation_data["conversation_end"].timetuple()) + 3600)

    technical_data = []
    db_averages = []
    data_averages = []
    average_tfidf, n = sparse.csr_matrix([]), 0

    time_windows = [900, 3600, 6*3600, 1500]  # 15 min, 60 min, 6h
    the_window = 1500
    for time_window in time_windows:
        if time_window != the_window:
            data_averages += common.get_averages_from_data(conversations, conversation_data["conversation_end"], time_window, currency, i, threshold=0.0, type="conversation", data_averages_only=True)
            technical_data.append(common.get_price_change(client, currency, date_from - time_window, date_from))
            technical_data.append(common.get_total_volume(client, currency, date_from - time_window, date_from) / common.get_total_volume(client, "all", date_from - time_window, date_from))
            technical_data.append(common.get_all_price_changes(client, date_from - time_window, date_from))

            db_averages += common.get_averages_from_db(client, conversation_data["conversation_end"], time_window, currency, conversations=False)
        else:
            _, average_tfidf, n, _ = common.get_averages_from_data(conversations, conversation_data["conversation_end"], time_window, currency, i, 0, type="conversation", data_averages_only=False)

    avg_reputation = []
    for message in conversation_data["messages"]:
        avg_reputation.append(get_user_reputation(client, message))

    _X = [1 / (n + 1), conversation_data["avg_sentiment"], conversation_data["avg_polarity"], 1 / (np.mean(avg_reputation) + 1), 1 / conversation_data["messages_len"]] + data_averages + db_averages + technical_data

    if not np.all(np.isfinite(_X)):
        return None

    tfidf = conversation_data["tfidf"] * (1 / (n + 1)) + (average_tfidf * (n / (n + 1))).multiply(conversation_data["tfidf"].power(0))
    tfidf = tfidf.multiply(weights)

    _X += tfidf.todense().tolist()[0]

    return sparse.csr_matrix(_X)


def create_Y(client, conversation_data, currency, window, margin):
    date_from = int(time.mktime(conversation_data["conversation_end"].timetuple()) + 3600)
    date_to = date_from + window
    percent_change = common.get_min_max_price_change(client, currency, date_from, date_to)

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

    for conversation in conversations:
        for currency in conversation["coin_mentions"]:
            if str(conversation["_id"]) + ":" + currency in ids:
                _Y = create_Y(client, conversation, currency, window, margin)
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

    for conversation in conversations:
        for currency in conversation["coin_mentions"]:
            if str(conversation["_id"]) + ":" + currency in ids:
                date_from = int(time.mktime(conversation["conversation_end"].timetuple()) + 3600)
                date_to = date_from + window
                _p = common.get_min_max_price_change(client, currency, date_from, date_to)
                if not np.isnan(_p):
                    p.append(_p)
                    currencies.append(currency.lower())
                else:
                    print("recompute IDs")
                    return None, None

    return p, currencies
