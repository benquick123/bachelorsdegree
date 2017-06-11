import pymongo
import requests
import json
import re


def get_currencies():
    url = "https://poloniex.com/public?command=returnCurrencies"
    res = requests.get(url)
    json_dict = json.loads(res.text)

    currencies = dict()

    for key, values in json_dict.items():
        currencies[key] = values["name"]
    return currencies

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["trollbox"]

curr = get_currencies()
currencies = set()

for key, value in curr.items():
    currencies.add(key.lower())
    currencies.add(value.lower())

print(currencies)

for entry in db.messages.find():
    text = entry["text"]
    text = re.sub('[^a-zA-Z0-9-_* ]', "", text)
    text = text.lower()
    text = set(text.split(" "))

    intersection = text.intersection(currencies)

    if len(intersection) > 0:
        db.messages.update_one({"_id": entry["_id"]}, {"$set": {"coin_mentions": list(intersection)}})
        print("UPDATE:", entry["_id"], "with coin_mentions:", intersection)
    #print(text)
