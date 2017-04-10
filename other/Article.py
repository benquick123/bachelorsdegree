from datetime import datetime
import pytz


class Article:
    def __init__(self, currency, date=datetime(2016, 1, 1, 0, 0, 0).replace(tzinfo=pytz.utc), title="", text="", authors=[], source=""):
        self.date = date
        self.text = text
        self.title = title
        self.authors = authors
        self.source = source
        self.currency = currency
        self.stemmed_text = ""
        self.stemmed_title = ""
