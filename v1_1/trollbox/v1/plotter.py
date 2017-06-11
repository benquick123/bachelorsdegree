import pymongo
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import operator

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db_prices = client["crypto_prices"]

currency = "xmr"
prices = np.array([])

for entry in db_prices[currency].find({"_id": {"$gte": 1472688000}}):
    if entry["_id"] % (24*3600) == 0:
        prices = np.append(prices, [entry["weightedAverage"]])

max_price = max(prices)
min_price = min(prices)
prices = (prices - min_price) / (max_price - min_price)

db_trollbox = client["trollbox"]

sell = dict()
buy = dict()

for entry in db_trollbox.messages.find({"date": {"$gte": "2016-09-01 00:00:00 UTC"}}):
    date = entry["date"].split(" ")[0]
    if date not in sell:
        sell[date] = 0
    if date not in buy:
        buy[date] = 0

    coin_mentions = set(entry["coin_mentions"])
    if currency in coin_mentions:
        words = set(entry["text"].split(" "))
        if "sell" in words:
            sell[date] += 1
        if "buy" in words:
            buy[date] += 1

buy = sorted(buy.items(), key=operator.itemgetter(0))
buy = np.array([value for key, value in buy])
buy = (buy - min(buy)) / (max(buy) - min(buy))

sell = sorted(sell.items(), key=operator.itemgetter(0))
sell = np.array([value for key, value in sell])
sell = (sell - min(sell)) / (max(sell) - min(sell))

plt.figure(figsize=(12, 9))
plt.plot(range(len(prices)), prices, "r-")
plt.plot(range(len(buy)), buy, "y-")
plt.plot(range(len(sell)), sell, "b-")
plt.show()
