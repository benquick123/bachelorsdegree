import pymongo
import v2.common as common
import re


def load_with_attr():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["trollbox"]

    conversations = []
    for conversation in db.conversations.find({"responses": True}, {"_id": 1, "conversation_end": 1, "coin_mentions": 1}):
        conversation_tree = build_tree(conversation["_id"], db)
        messages = get_messages(conversation_tree)
        conversation["messages"] = messages
        conversations.append(conversation)

    return conversations


def load_without_attr(save_to_db=True):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["trollbox"]

    pattern = re.compile("[\W_]+")

    conversations = []
    for conversation in db.conversations.find({"responses": True}, {"_id": 1, "conversation_end": 1, "coin_mentions": 1}):
        conversation_tree = build_tree(conversation["_id"], db)

        messages = []
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

                if not common.is_stop_word(word):
                    text.append(word)

            sentiment, polarity = common.calc_sentiment_polarity(text)
            text = common.stem(text)

            message["clean_text"] = text
            message["sentiment"] = sentiment
            message["polarity"] = polarity

            if save_to_db:
                db.messages.update_one({"_id", message["_id"]}, {"$set": {"clean_text": text, "sentiment": sentiment, "polarity": polarity}})

            if len(text) > 1:
                messages.append(message)

        conversation["messages"] = messages
        conversations.append(conversation)

    return conversations


def build_tree(_id, db):
    tree = db.messages.find_one({"_id": _id}, {"_id": 1, "user_id": 1, "date": 1, "text": 1, "responses": 1, "sentiment": 1, "polarity": 1, "clean_text": 1})

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
    if word[-1] == ",":
        return None
    return word


def create_matrix(conversations):
    X = []
    Y = []
    for conversation in conversations:
        _X, _Y = create_matrix_line(conversation)
        X.append(_X)
        Y.append(_Y)


def create_matrix_line(conversation_data):
    X = []
    Y = 0
    return X, Y
