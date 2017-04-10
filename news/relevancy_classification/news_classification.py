import pymongo
import random


def random_sample():
    ids = db.articles.distinct("_id", {})

    samples = random.sample(range(len(ids)), 1000)

    for sample in samples:
        _id = ids[sample]
        article = db.articles.find_one({"_id": _id})
        print(article["title"], "______", _id)
        b = input("Relevant? ")
        _b = True if b == "1" else False
        db.articles.update({"_id": _id}, {"$set": {"relevant": _b}})
        print("Successfully updated with relevancy: " + str(_b))


def daily_updates():
    pass

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["news"]

random_sample()
