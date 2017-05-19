from datetime import datetime


class Tweet:
    currencies_mapper = {}

    def __init__(self, tweet_id, currencies=list(), date=datetime(1970, 1, 1, 0, 0, 0), text="", author_id=""):
        self.id = tweet_id
        self.date = date
        self.text = text
        self.stemmed_text = ""
        self.author = author_id
        self.currencies = currencies

        self.sentiment_text = ""

    def map_currencies(self):
        for currency in self.currencies:
            if currency not in type(self).currencies_mapper.keys():
                type(self).currencies_mapper[currency] = len(type(self).currencies_mapper)

    def __str__(self):
        string = ""
        string += "currency: " + self.currencies[0] + ", date: " + self.date.strftime('%Y-%m-%d %H:%M:%S') + ", id: " + self.id + "\n"
        string += "stemmed text[:10]: " + " ".join(self.stemmed_text[:10]) + "; (" + str(self.sentiment_text).strip("{}") + ")"
        return string

