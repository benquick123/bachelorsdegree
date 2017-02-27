from datetime import datetime
import pytz


class Event:
    def __init__(self):
        self.articles = []
        self.first_date = datetime(2016, 1, 1, 0, 0, 0).replace(tzinfo=pytz.utc)
        self.max_price = int("-inf")
        self.min_price = int("inf")
        self.time_to_max = 0
        self.time_to_min = 0

        self.bow = set()
        self.keywords = set()
        self.meta_weight = 1
        self.weight = 1
