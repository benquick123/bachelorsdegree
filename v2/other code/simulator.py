import pymongo
import v2.twitter as twitter
import v2.trollbox as trollbox
import v2.news as news


def simulate(ids, window, type, predictions):
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db = client["crypto_prices"]
    currencies = set(db.collection_names())

    earnings = {currency: 1 for currency in currencies}
    if type == "articles":
        p, c = news.get_percent_currencies(ids, window)
    elif type == "conversations":
        p, c = [], []
        exit()
    elif type == "tweets":
        p, c = [], []
        exit()
    else:
        p, c = [], []
        exit()

    for i, prediction in enumerate(predictions):
        if prediction == 1 and p[i] > 0:
            earnings[c[i]] *= (1 + p[i])
        elif prediction == 1 and p[i] <= 0:
            earnings[c[i]] *= (1 / (1 + abs(p[i])))
        elif prediction == -1 and p[i] < 0:
            earnings[c[i]] *= (1 + abs(p[i]))
        elif prediction == -1 and p[i] >= 0:
            earnings[c[i]] *= (1 / (1 + p[i]))

    return sum([earning for earning in earnings.values()]) / len(earnings)
