import pymongo
from importlib import import_module
from datetime import datetime
from v1_1.trollbox.v1.response_tree import build_tree, print_tree

# build_tree = import_module("response_tree").build_tree
# print_tree = import_module("response_tree").print_tree


def find_conversation_end(tree):
    date = datetime.strptime(tree["date"], "%Y-%m-%d %H:%M:%S UTC")
    responses = tree["responses"]
    for response in responses:
        ret_date = find_conversation_end(response)
        if ret_date > date:
            date = ret_date

    return date


def all_mentioned_currencies(tree):
    currencies = set()
    currencies = currencies.union(set(tree["coin_mentions"]))

    responses = tree["responses"]
    for response in responses:
        currencies = currencies.union(all_mentioned_currencies(response))
    return currencies


def build_conversation_database(messages, db):
    for _id, responses in messages.items():
        tree = build_tree(_id, db, print=False)

        conversation_end = find_conversation_end(tree)
        conversation_start = datetime.strptime(tree["date"], "%Y-%m-%d %H:%M:%S UTC")

        mentioned_currencies = list(all_mentioned_currencies(tree))
        sentiment = -1
        subjectivity = -1
        context = []

        data = {
            "_id": _id,
            "responses": responses,
            "conversation_start": conversation_start,
            "conversation_end": conversation_end,
            "coin_mentions": mentioned_currencies,
            "sentiment": sentiment,
            "subjectivity": subjectivity,
            "context": context
        }

        db.conversations.insert_one(data)
        print("SUCCESS:", _id, "inserted into db.conversations.")


def __main__():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["trollbox"]

    all_messages = dict()
    to_remove = set()

    for entry in db.messages.find():
        all_messages[entry["_id"]] = len(entry["responses"]) > 0
        to_remove = to_remove.union(set(entry["responses"]))
        if entry["_id"] % 10000 == 0:
            to_remove_tmp = set()
            print("to_remove size before is", len(to_remove), end="")
            for entry_to_remove in to_remove:
                if entry_to_remove in all_messages:
                    del all_messages[entry_to_remove]
                    to_remove_tmp.add(entry_to_remove)
            to_remove -= to_remove_tmp

            print(" and after", len(to_remove), "UPDATED at ID:", str(entry["_id"]))

    build_conversation_database(all_messages, db)

# __main__()
