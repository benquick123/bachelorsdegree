import pymongo
import poloniex
import time
from datetime import datetime
import requests
import json

def setupPoloniex():
    file = open("../credentials/james.champ.at.tuta.io.key", "r")
    key = file.readline()[0:-1]
    secret = file.readline()[0:-1]
    file.close()
    polo = poloniex.Poloniex(key, secret, timeout=10)
    return polo

def saveChartData(db, polo, currencies):
    end = time.mktime(datetime.strptime("01.11.2016", "%d.%m.%Y").timetuple())
    start = time.mktime(datetime.strptime("31.10.2015", "%d.%m.%Y").timetuple())
    period = "300"

    for currency in currencies:
        currencyPair = "BTC_" + currency
        if currency == "BTC":
            currencyPair = "USDT_BTC"

        results = polo.api("returnChartData", {"currencyPair": currencyPair, "period": period, "end": end, "start": start})

        if len(results) > 1:
            for result in results:
                result["_id"] = result["date"]
                del result["date"]
                try:
                    db[currency.lower()].insert_one(result)
                except pymongo.errors.DuplicateKeyError:
                    print("ERROR: key", result["_id"], "already exists.")
            print("SUCCESS: saved", currency + ".")

        else:
            if "error" in results:
                print("ERROR:", currencyPair, results["error"])
            else:
                print("ERROR:", currencyPair, results)

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["crypto_prices"]


last = "NXTI"
polo = setupPoloniex()
currencies = polo.api("returnCurrencies")

to_delete = set()
for currency in currencies.keys():
    if currency < last:
        to_delete.add(currency)

for currency in to_delete:
    del currencies[currency]

currencies = sorted(list(currencies.keys()))

saveChartData(db, polo, currencies)
