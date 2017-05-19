import time
from datetime import datetime
import pymongo
import pytz
import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn

import news.functions as f_news
import twitter.functions as f_twitter
from Event import Event


def create_events(currencies, date_from, date_to, db, plot=False):
    events = dict()

    time_start = int(time.mktime(date_from.timetuple()) + 3600)     # +1h for time zone adjustment
    time_end = int(time.mktime(date_to.timetuple()) + 3600)         # +1h for time zone adjustment

    avg_window_size = 11 * 72       # 11 samples takes one hour
    std_window_size = 11 * 72       # 72h = 3d

    for currency in currencies:
        print(currency)

        prices = []
        price_values = []

        for price in db[currency].find({"$and": [{"_id": {"$gte": time_start}}, {"_id": {"$lte": time_end}}]}).sort("_id", 1):
            prices.append(price)
            price_values.append(price["weightedAverage"])

        moving_average = pandas.rolling_mean(np.array(price_values), window=avg_window_size)
        moving_std = pandas.rolling_std(np.array(price_values), window=std_window_size)

        events_min_max = []
        curr_event_i = np.nan
        curr_event_price = np.nan
        prev_event_type = np.nan
        in_event = False
        in_wait = False

        for i in range(len(moving_average)):
            # end event or continue
            if in_event and not in_wait:
                in_wait = True
                in_event = False

            if in_wait and ((prev_event_type == "up" and moving_average[i] > price_values[i]) or (
                    prev_event_type == "down" and moving_average[i] < price_values[i])):
                in_wait = False
                events_min_max.append({"i": curr_event_i, "price": curr_event_price, "event_type": prev_event_type})
                curr_event_i = np.nan
                curr_event_price = np.nan
                prev_event_type = np.nan

            if np.isnan(moving_average[i]) or np.isnan(moving_std[i]):
                continue
            elif price_values[i] > moving_average[i]+moving_std[i]:
                # positive event
                in_event = True
                in_wait = False
                if np.isnan(curr_event_price) or curr_event_price < price_values[i]:
                    curr_event_i = i
                    curr_event_price = price_values[i]
                    prev_event_type = "up"
            elif price_values[i] < moving_average[i]-moving_std[i]:
                # negative event
                in_event = True
                in_wait = False
                if np.isnan(curr_event_price) or curr_event_price > price_values[i]:
                    curr_event_i = i
                    curr_event_price = price_values[i]
                    prev_event_type = "down"

        if len(events_min_max) > 0:
            events_min_max.pop(0)

        p_values = []

        for event in events_min_max:
            end = event["i"]
            event_type = event["event_type"]

            i = end-1
            while True:
                if (event_type == "up" and moving_average[i] > price_values[i]) or (event_type == "down" and moving_average[i] < price_values[i]):
                    if i != end:
                        # create event
                        first_price = prices[i]
                        last_price = prices[end]

                        event_id = currency + ":" + str(first_price)
                        _event = Event(event_id=event_id, currency=currency)
                        _event.first_price_info = first_price
                        _event.last_price_info = last_price
                        _event.p = (last_price["weightedAverage"] - first_price["weightedAverage"]) / first_price["weightedAverage"]
                        _event.t = last_price["_id"] - first_price["_id"]
                        _event.magnitude = (last_price["weightedAverage"] - moving_average[end]) / moving_std[end]
                        _event.time_x1 = i
                        _event.time_x2 = end
                        events[event_id] = _event

                        p_values.append(_event.p)
                    break
                i -= 1

        p_mean = np.mean(p_values)
        p_std = np.std(p_values)
        for event in events.values():
            event.magnitude_p = event.p / p_std

        if plot:
            plt.plot(range(len(price_values)), price_values, "r-")
            plt.plot(range(len(moving_average)), moving_average, "b-")
            plt.fill_between(range(len(price_values)), np.array(moving_average) + moving_std, np.array(moving_average) - moving_std, alpha=0.7)
            for event in events.values():
                plt.axvspan(event.time_x1, event.time_x2, color="green", alpha=abs(event.magnitude_p / 2) if abs(event.magnitude_p / 2) <= 1 else 1)
            # plt.scatter([x["i"] for x in events_min_max], [y["price"] for y in events_min_max])
            plt.show()

    sorted_keys = sorted(events.keys())
    return events, sorted_keys


def save_events(db, events, sorted_keys):
    for key in sorted_keys:
        event = events[key].get_data()
        event["_id"] = key
        db.events.insert_one(event)


def load_events(db, p_threshold=None):
    events = dict()
    sorted_keys = []
    for event in db.events.find({}):
        key = event["_id"]
        if p_threshold is None or abs(event["magnitude_p"]) > p_threshold:
            sorted_keys.append(key)
            _event = Event(key)
            _event.set_data(event)
            events[key] = _event

    sorted_keys = sorted(sorted_keys)
    return events, sorted_keys


def load_articles(db, currency, date_start, date_end):
    return f_news.load_articles(db, currency, date_start, date_end, p=True)


def load_articles_from_ids(db, article_ids):
    return f_news.load_articles_from_ids(db, article_ids)


def load_tweets(db, currency, date_start, date_end, tweets=None):
    return f_twitter.load_tweets(db, currency, date_start, date_end, _tweets=tweets)


def load_tweets_from_ids(db, tweet_ids):
    return f_twitter.load_tweets_from_ids(db, tweet_ids)


def load_conversations(db, date_start, date_end):
    conversations = dict()
    return conversations


def __main__():
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    db_prices = client["crypto_prices"]
    db_events = client["events"]

    currencies = db_prices.collection_names()

    date_from = datetime(2015, 11, 1, 0, 0, 0).replace(tzinfo=pytz.utc)
    date_to = datetime(2016, 10, 31, 23, 59, 59).replace(tzinfo=pytz.utc)

    # call only once to save events to db
    # events, sorted_keys = create_events(currencies, date_from, date_to, db_prices, plot=False)
    # save_events(db_events, events, sorted_keys)

    # for now, i only load events with % change > 0.5 std
    events, sorted_keys = load_events(db_events, p_threshold=0.5)
    print(len(events))

    db_news = client["news"]
    db_twitter = client["twitter"]
    db_trollbox = client["trollbox"]

    last_curr = ""
    tweets = []

    for key in sorted_keys:
        event = events[key]
        date_start = datetime.utcfromtimestamp(event.first_price_info["_id"])
        date_end = datetime.utcfromtimestamp(event.last_price_info["_id"])
        currency = event.currency

        # save article ids to db. only once.
        # event.articles = load_articles(db_news, currency, date_start, date_end)
        # article_ids = []
        # for article in event.articles:
        #     article_ids.append(article.id)
        # db_events.events.update_one({"_id": key}, {"$set": {"article_ids": article_ids}})

        event.articles = load_articles_from_ids(db_news, event.article_ids)

        # save twitter ids to db. only once
        # if last_curr != currency:
        #     last_curr = currency
        #     tweets = []
        #     for tweet in db_twitter.messages.find({"crypto_currency": currency.upper()}):
        #         tweets.append(tweet)

        # event.tweets = load_tweets(db_twitter, currency, date_start, date_end, tweets=tweets)
        # tweet_ids = []
        # for tweet in event.tweets:
        #     tweet_ids.append(tweet.id)
        # db_events.events.update_one({"_id": key}, {"$set": {"tweet_ids": tweet_ids}})

        event.tweets = load_tweets_from_ids(db_twitter, event.tweet_ids)
        # print(event)

        event.conversations = load_conversations(db_trollbox, date_start, date_end)


__main__()


"""
    for currency in currencies:
        window_prices = []
        last_event_id = None

        for price in db[currency].find({"$and": [{"_id": {"$gte": time_start}}, {"_id": {"$lte": time_end}}]}).sort("_id", 1):
            if len(window_prices) == 0 or window_prices[-1]["_id"]-window_prices[0]["_id"] < T:
                # first, fill array to fit time window
                window_prices.append(price)

            else:
                # second, find out if price change in time window was above threshold
                window_prices.append(price)
                max_in_array = max(window_prices[0:], key=lambda x: x["weightedAverage"])
                min_in_array = min(window_prices[0:], key=lambda x: x["weightedAverage"])

                try:
                    first_price = window_prices[0] if last_event_id is None else events[last_event_id].first_price_info
                    # compute relative changes
                    pmax = (max_in_array["weightedAverage"] - first_price["weightedAverage"]) / first_price["weightedAverage"]
                    pmin = (min_in_array["weightedAverage"] - first_price["weightedAverage"]) / first_price["weightedAverage"]

                    if (abs(pmax) >= threshold or abs(pmin) >= threshold) and (last_event_id is None or window_prices[0]["_id"]-events[last_event_id].max_price_info["_id"] < T or window_prices[0]["_id"]-events[last_event_id].min_price_info["_id"] < T):
                        # create / append new info to event
                        if last_event_id is None or len(events) == 0:
                            # create new event
                            event = Event(currency, first_price)
                            event.max_price_info = max_in_array
                            event.min_price_info = min_in_array
                            event.last_relevant_price_info = first_price
                            event.last_price_info = first_price

                            event_start = first_price["_id"]
                            last_event_id = currency + ":" + str(event_start)
                            events[last_event_id] = event

                        elif last_event_id is not None:
                            # check if current max and min are in prices array. if not, end this event, otherwise append new info.
                            if window_prices.count(events[last_event_id].max_price_info) < 1 and window_prices.count(events[last_event_id].min_price_info) < 1:
                                events[last_event_id].pmax = (events[last_event_id].max_price_info["weightedAverage"] - events[last_event_id].first_price_info["weightedAverage"]) / events[last_event_id].first_price_info["weightedAverage"]
                                events[last_event_id].pmin = (events[last_event_id].min_price_info["weightedAverage"] - events[last_event_id].first_price_info["weightedAverage"]) / events[last_event_id].first_price_info["weightedAverage"]
                                events[last_event_id].tmax = events[last_event_id].max_price_info["_id"] - events[last_event_id].first_price_info["_id"]
                                events[last_event_id].tmin = events[last_event_id].min_price_info["_id"] - events[last_event_id].first_price_info["_id"]
                                events[last_event_id].classification = 0 if abs(events[last_event_id].pmax) <= abs(events[last_event_id].pmin) else 1
                                last_event_id = None
                            else:
                                # append to last event
                                events[last_event_id].last_price_info = window_prices[0]                 # debugging
                                if max_in_array["weightedAverage"] > events[last_event_id].max_price_info["weightedAverage"]:
                                    events[last_event_id].last_relevant_price_info = window_prices[0]
                                    events[last_event_id].max_price_info = max_in_array
                                if min_in_array["weightedAverage"] < events[last_event_id].min_price_info["weightedAverage"]:
                                    events[last_event_id].last_relevant_price_info = window_prices[0]
                                    events[last_event_id].min_price_info = min_in_array

                    else:
                        # correct pmin, pmax, tmin, tmax values
                        if last_event_id is not None:
                            events[last_event_id].pmax = (events[last_event_id].max_price_info["weightedAverage"] - events[last_event_id].first_price_info["weightedAverage"]) / events[last_event_id].first_price_info["weightedAverage"]
                            events[last_event_id].pmin = (events[last_event_id].min_price_info["weightedAverage"] - events[last_event_id].first_price_info["weightedAverage"]) / events[last_event_id].first_price_info["weightedAverage"]
                            events[last_event_id].tmax = events[last_event_id].max_price_info["_id"] - events[last_event_id].first_price_info["_id"]
                            events[last_event_id].tmin = events[last_event_id].min_price_info["_id"] - events[last_event_id].first_price_info["_id"]
                            events[last_event_id].classification = 0 if abs(events[last_event_id].pmax) <= abs(events[last_event_id].pmin) else 1
                        last_event_id = None

                    del window_prices[0]
                except ZeroDivisionError:
                    pass
"""