from datetime import datetime
import pytz


class Article:
    currencies_mapper = {}

    def __init__(self, article_id, currencies=[], date=datetime(1970, 1, 1, 0, 0, 0), title="", text="", authors=[], source=""):
        self.id = article_id
        self.date = date
        self.text = text
        self.title = title
        self.authors = authors
        self.source = source
        self.currencies = currencies
        self.stemmed_text = ""
        self.stemmed_title = ""

        self.sentiment_text = {}
        self.sentiment_title = {}

        self.cluster = -1

    def map_currencies(self):
        for currency in self.currencies:
            if currency not in type(self).currencies_mapper.keys():
                type(self).currencies_mapper[currency] = len(type(self).currencies_mapper)

    def get_article_matrix(self):
        # currency, text_pos, text_neu, text_neg, text_comb, title_pos, title_neu, title_neg, title_comb, classification
        param_lines = []
        for i, currency in enumerate(self.currencies):
            # _id = currency + ":" + ":".join(self.id.split(":")[1:])
            param_lines.append([type(self).currencies_mapper[currency], self.sentiment_text["pos"], self.sentiment_text["neu"], self.sentiment_text["neg"], self.sentiment_text["compound"], self.sentiment_title["pos"], self.sentiment_title["neu"], self.sentiment_title["neg"], self.sentiment_title["compound"], self.classification[i]])
        return param_lines

    def __str__(self):
        string = ""
        string += "currency: " + self.currencies[0] + ", date: " + self.date.strftime('%Y-%m-%d %H:%M:%S') + ", id: " + self.id + "\n"
        string += "stemmed title: " + " ".join(self.stemmed_title) + "; (" + str(self.sentiment_title).strip("{}") + ")\n"
        string += "stemmed text[:10]: " + " ".join(self.stemmed_text[:10]) + "; (" + str(self.sentiment_text).strip("{}") + ")"
        return string
