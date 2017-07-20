import pymongo
import numpy as np

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["crypto_prices"]

collections = set(db.collection_names())
if "all" in collections:
    collections.remove("all")

prices = dict()
for currency in collections:
    prices[currency] = [0, np.nan]

start = 1446246000
end = 1477954800

while start <= end:
    volumes = []
    for currency in collections:
        price_info = db[currency].find_one({"_id": start})
        if price_info is not None:
            if prices[currency][0] == 0:
                prices[currency][0] = 1
            else:
                prices[currency][0] *= (price_info["weightedAverage"] / prices[currency][1])
            prices[currency][1] = price_info["weightedAverage"]
            volumes.append(price_info["volume"])
        else:
            prices[currency][0] = 0

    print("volume:", sum(volumes))
    print("price:", sum([item[0] for item in prices.values()]))
    db["all"].insert_one({"_id": start, "volume": sum(volumes), "weightedAverage": sum([item[0] for item in prices.values()])})
    start += 300
