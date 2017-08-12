import numpy as np
import pymongo

import v2.common as common
import time
from scipy import sparse

def best_back_windows(**kwargs):
    print("FIND BEST BACK WINDOWS")
    n_iter = kwargs["n_iter"]
    n_iter = 100
    back_window_range = kwargs["back_window_range"]
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    ids = kwargs["IDs"]
    raw_data = kwargs["raw_data"]
    type = kwargs["type"]
    final_set_f = kwargs["final_set_f"]
    del kwargs

    ids, _, _ = set(final_set_f(None, None, ids))

    is_conversation = False
    date_key = ""
    currency_key = ""
    articles = conversations = tweets = True
    if type == "articles":
        date_key = "date"
        currency_key = "currency"
        articles = False
    elif type == "conversations":
        is_conversation = True
        date_key = "conversation_end"
        currency_key = "coin_mentions"
        conversations = False
    elif type == "tweets":
        date_key = "posted_time"
        currency_key = "crypto_currency"
        tweets = False

    i = 0
    price_X = volume_X = price_all_X = distribution_1_X = distribution_2_X = distribution_3_X = sentiment_1_X = sentiment_2_X = sentiment_3_X = polarity_1_X = polarity_2_X = polarity_3_X = None
    while i < n_iter:
        back_window = 300 * round((back_window_range[0] + (i+1 / n_iter) * (back_window_range[1] - back_window_range[0])) / 300)
        _price_X = _volume_X = _price_all_X = _distribution_X = _sentiment_X = _polarity_X = []
        if not is_conversation:
            for i, text in enumerate(raw_data):
                if text["_id"] in ids:
                    averages = []
                    averages += common.get_averages_from_data(raw_data, text[date_key], back_window, text[currency_key], i, threshold=0.0, type=type[:-1], data_averages_only=True)[0]
                    averages += common.get_averages_from_db(client, text[date_key], back_window, text[currency_key], articles=articles, conversations=conversations, tweets=tweets)

                    _distribution_X.append([distribution for i, distribution in enumerate(averages) if i % 3 == 0])
                    _polarity_X.append([polarity for i, polarity in enumerate(averages) if i % 3 == 1])
                    _sentiment_X.append([sentiment for i, sentiment in enumerate(averages) if i % 3 == 2])

                    date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                    _price_X.append(common.get_price_change(client, text[currency_key], date_from - back_window, date_from))
                    _volume_X.append(common.get_total_volume(client, text[currency_key], date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from))
                    _price_all_X.append(common.get_all_price_changes(client, date_from - back_window, date_from))
        else:
            for i, text in enumerate(raw_data):
                for currency in text[currency_key]:
                    if str(text["_id"]) + ":" + currency in ids:
                        averages = []
                        averages += common.get_averages_from_data(raw_data, text[date_key], back_window, currency, i, threshold=0.0, type=type[:-1], data_averages_only=True)[0]
                        averages += common.get_averages_from_db(client, text[date_key], back_window, currency, articles=articles, conversations=conversations, tweets=tweets)

                        _distribution_X.append([distribution for i, distribution in enumerate(averages) if i % 3 == 0])
                        _polarity_X.append([polarity for i, polarity in enumerate(averages) if i % 3 == 1])
                        _sentiment_X.append([sentiment for i, sentiment in enumerate(averages) if i % 3 == 2])

                        date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                        _price_X.append(common.get_price_change(client, currency, date_from - back_window, date_from))
                        _volume_X.append(common.get_total_volume(client, currency, date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from))
                        _price_all_X.append(common.get_all_price_changes(client, date_from - back_window, date_from))

        price_X = sparse.csr_matrix(_price_X) if price_X is None else sparse.vstack([price_X, sparse.csr_matrix(_price_X)]).tocsr()
        volume_X = sparse.csr_matrix(_volume_X) if volume_X is None else sparse.vstack([volume_X, sparse.csr_matrix(_volume_X)]).tocsr()
        price_all_X = sparse.csr_matrix(_price_all_X) if price_all_X is None else sparse.vstack([price_all_X, sparse.csr_matrix(_price_all_X)]).tocsr()

        distribution_1_X = sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 0]) if distribution_1_X is None else sparse.vstack([distribution_1_X, sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 0])]).tocsr()
        distribution_2_X = sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 1]) if distribution_2_X is None else sparse.vstack([distribution_2_X, sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 1])]).tocsr()
        distribution_3_X = sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 2]) if distribution_3_X is None else sparse.vstack([distribution_3_X, sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 2])]).tocsr()

        sentiment_1_X = sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 0]) if sentiment_1_X is None else sparse.vstack([sentiment_1_X, sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 0])]).tocsr()
        sentiment_2_X = sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 1]) if sentiment_2_X is None else sparse.vstack([sentiment_2_X, sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 1])]).tocsr()
        sentiment_3_X = sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 2]) if sentiment_3_X is None else sparse.vstack([sentiment_3_X, sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 2])]).tocsr()

        polarity_1_X = sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 0]) if polarity_1_X is None else sparse.vstack([polarity_1_X, sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 0])])
        polarity_2_X = sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 1]) if polarity_2_X is None else sparse.vstack([polarity_2_X, sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 1])])
        polarity_3_X = sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 2]) if polarity_3_X is None else sparse.vstack([polarity_3_X, sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 2])])

        i += 1