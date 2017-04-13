from datetime import datetime
import pytz


class Article:
    currencies_mapper = {}

    def __init__(self, id, currencies=[], date=0, title="", text="", authors=[], source=""):
        self.id = id
        self.date = date
        self.text = text
        self.title = title
        self.authors = authors
        self.source = source
        self.currencies = currencies
        self.stemmed_text = ""
        self.stemmed_title = ""

        self.pmax = []
        self.pmin = []
        self.tmax = []
        self.tmin = []
        self.classification = []

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
