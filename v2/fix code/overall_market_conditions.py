import pymongo
import numpy as np

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["crypto_prices"]

collections = set(db.collection_names())
if "all" in collections:
    collections.remove("all")

prev_prices = dict()
for currency in collections:
    prev_prices[currency] = np.nan

start = 1446246000
end = 1477954800

while start <= end:
    volumes = []
    prices = []
    for currency in collections:
        price_info = db[currency].find_one({"_id": start})
        if price_info is not None:
            if np.isnan(prev_prices[currency]):
                prices.append(1)
            else:
                prices.append(price_info["weightedAverage"] / prev_prices[currency])
            volumes.append(price_info["volume"])
            prev_prices[currency] = price_info["weightedAverage"] if price_info["weightedAverage"] != 0 else np.nan
        else:
            prev_prices[currency] = np.nan

    print(start)
    db["all"].insert_one({"_id": start, "volume": sum(volumes), "weightedAverage": np.mean(prices)})
    start += 300
