from datetime import datetime
import pytz


class Event:
    def __init__(self, event_id, currency=""):
        self.id = event_id
        self.currency = currency
        self.first_price_info = {}
        self.last_price_info = {}
        self.time_x1 = 0
        self.time_x2 = 0

        self.p = 0
        self.t = 0
        self.magnitude = 0
        self.magnitude_p = 0

        self.articles = []
        self.article_ids = []
        self.tweets = []
        self.tweet_ids = []
        self.conversations = []
        self.conversation_ids = []

        self.related_events = []

    def __str__(self):
        string = ""
        string += "currency: " + self.currency + "\n"
        string += "p: " + str(self.p) + " (" + datetime.utcfromtimestamp(self.last_price_info["_id"]).strftime('%Y-%m-%d %H:%M:%S') + "), magnitude_p: " + str(self.magnitude_p) + ", magnitude: " + str(self.magnitude) + "\n"
        string += "start: " + datetime.utcfromtimestamp(self.first_price_info["_id"]).strftime('%Y-%m-%d %H:%M:%S') + ", end: " + datetime.utcfromtimestamp(self.last_price_info["_id"]).strftime('%Y-%m-%d %H:%M:%S') + "\n"
        string += "num articles: " + str(len(self.articles)) + "\n"
        string += "num tweets: " + str(len(self.tweets)) + "\n"
        string += "num conversations: " + str(len(self.conversations))
        return string

    def get_data(self):
        d = dict()
        d["_id"] = self.id
        d["currency"] = self.currency
        d["first_price_info"] = self.first_price_info
        d["last_price_info"] = self.last_price_info
        d["p"] = self.p
        d["t"] = self.t
        d["magnitude"] = self.magnitude
        d["magnitude_p"] = self.magnitude_p

        d["article_ids"] = self.article_ids
        d["tweet_ids"] = self.tweet_ids
        d["conversation_ids"] = self.conversation_ids

        d["time_x"] = (self.time_x1, self.time_x2)
        return d

    def set_data(self, data):
        self.id = data["_id"]
        self.currency = data["currency"]
        self.first_price_info = data["first_price_info"]
        self.last_price_info = data["last_price_info"]
        self.p = data["p"]
        self.t = data["t"]
        self.magnitude = data["magnitude"]
        self.magnitude_p = data["magnitude_p"]

        self.time_x1, self.time_x2 = data["time_x"]

        if "article_ids" in data:
            self.article_ids = data["article_ids"]
        if "tweet_ids" in data:
            self.tweet_ids = data["tweet_ids"]
        if "conversation_ids" in data:
            self.conversation_ids = data["conversation_ids"]
