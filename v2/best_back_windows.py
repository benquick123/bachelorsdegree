import numpy as np
import pymongo

import common as common
import time
from scipy import sparse
from sklearn.cluster import KMeans
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat


def parallelized_matrix_creation(k, n_iter, back_window_range, raw_data, ids, type, date_key, currency_key, client, is_conversation, articles, conversations, tweets):
    back_window = 300 * round((back_window_range[0] + (k + 1 / n_iter) * (back_window_range[1] - back_window_range[0])) / 300)
    _price_X = _volume_X = _price_all_X = _distribution_X = _sentiment_X = _polarity_X = []

    print("thread", k)

    if not is_conversation:
        for i, text in enumerate(raw_data):
            if text["_id"] in ids:
                averages = []
                averages += common.get_averages_from_data(raw_data, text[date_key], back_window, text[currency_key], i, threshold=0.0, type=type[:-1], data_averages_only=True)
                averages += common.get_averages_from_db(client, text[date_key], back_window, text[currency_key], articles=articles, conversations=conversations, tweets=tweets)

                _distribution_X.append([distribution for j, distribution in enumerate(averages) if j % 3 == 0])
                _polarity_X.append([polarity for j, polarity in enumerate(averages) if j % 3 == 1])
                _sentiment_X.append([sentiment for j, sentiment in enumerate(averages) if j % 3 == 2])

                date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                _price_X.append(common.get_price_change(client, text[currency_key], date_from - back_window, date_from))
                _volume_X.append(common.get_total_volume(client, text[currency_key], date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from))
                _price_all_X.append(common.get_all_price_changes(client, date_from - back_window, date_from))
    else:
        for i, text in enumerate(raw_data):
            for currency in text[currency_key]:
                if str(text["_id"]) + ":" + currency in ids:
                    averages = []
                    averages += common.get_averages_from_data(raw_data, text[date_key], back_window, currency, i, threshold=0.0, type=type[:-1], data_averages_only=True)
                    averages += common.get_averages_from_db(client, text[date_key], back_window, currency, articles=articles, conversations=conversations, tweets=tweets)

                    _distribution_X.append([distribution for j, distribution in enumerate(averages) if j % 3 == 0])
                    _polarity_X.append([polarity for j, polarity in enumerate(averages) if j % 3 == 1])
                    _sentiment_X.append([sentiment for j, sentiment in enumerate(averages) if j % 3 == 2])

                    date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                    _price_X.append(common.get_price_change(client, currency, date_from - back_window, date_from))
                    _volume_X.append(common.get_total_volume(client, currency, date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from))
                    _price_all_X.append(common.get_all_price_changes(client, date_from - back_window, date_from))

    price_X = sparse.csr_matrix(_price_X)
    volume_X = sparse.csr_matrix(_volume_X)
    price_all_X = sparse.csr_matrix(_price_all_X)

    distribution_1_X = sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 0])
    distribution_2_X = sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 1])
    distribution_3_X = sparse.csr_matrix([distribution for j, distribution in enumerate(_distribution_X) if j % 3 == 2])

    sentiment_1_X = sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 0])
    sentiment_2_X = sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 1])
    sentiment_3_X = sparse.csr_matrix([sentiment for j, sentiment in enumerate(_sentiment_X) if j % 3 == 2])

    polarity_1_X = sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 0])
    polarity_2_X = sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 1])
    polarity_3_X = sparse.csr_matrix([polarity for j, polarity in enumerate(_polarity_X) if j % 3 == 2])

    return price_X, volume_X, price_all_X, distribution_1_X, distribution_2_X, distribution_3_X, sentiment_1_X, sentiment_2_X, sentiment_3_X, polarity_1_X, polarity_2_X, polarity_3_X, back_window


def best_back_windows(**kwargs):
    print("FIND BEST BACK WINDOWS")
    n_iter = kwargs["n_iter"]
    n_iter = 100
    back_window_range = kwargs["back_window_range"]
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    ids = np.array(kwargs["IDs"])
    raw_data = kwargs["raw_data"]
    type = kwargs["type"]
    final_set_f = kwargs["final_set_f"]
    del kwargs

    ids = set(final_set_f(None, None, ids)[0])

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

    price_X = volume_X = price_all_X = distribution_1_X = distribution_2_X = distribution_3_X = sentiment_1_X = sentiment_2_X = sentiment_3_X = polarity_1_X = polarity_2_X = polarity_3_X = None
    windows = []

    pool = ThreadPool()
    results = pool.starmap(parallelized_matrix_creation, zip(list(range(n_iter)), repeat(n_iter), repeat(back_window_range), repeat(raw_data), repeat(ids), repeat(type), repeat(date_key), repeat(currency_key), repeat(client), repeat(is_conversation), repeat(articles), repeat(conversations), repeat(tweets)))
    pool.close()
    pool.join()

    for result in results:
        price_X = sparse.csr_matrix(result[0]) if price_X is None else sparse.vstack([price_X, result[0]])
        volume_X = sparse.csr_matrix(result[1]) if volume_X is None else sparse.vstack([volume_X, result[1]])
        price_all_X = sparse.csr_matrix(result[2]) if price_all_X is None else sparse.vstack([price_all_X, result[2]])

        distribution_1_X = sparse.csr_matrix(result[3]) if distribution_1_X is None else sparse.vstack([distribution_1_X, result[3]])
        distribution_2_X = sparse.csr_matrix(result[4]) if distribution_2_X is None else sparse.vstack([distribution_2_X, result[4]])
        distribution_3_X = sparse.csr_matrix(result[5]) if distribution_3_X is None else sparse.vstack([distribution_3_X, result[5]])

        sentiment_1_X = sparse.csr_matrix(result[6]) if sentiment_1_X is None else sparse.vstack([sentiment_1_X, result[6]])
        sentiment_2_X = sparse.csr_matrix(result[7]) if sentiment_2_X is None else sparse.vstack([sentiment_2_X, result[7]])
        sentiment_3_X = sparse.csr_matrix(result[8]) if sentiment_3_X is None else sparse.vstack([sentiment_3_X, result[8]])

        polarity_1_X = sparse.csr_matrix(result[9]) if polarity_1_X is None else sparse.vstack([polarity_1_X, result[9]])
        polarity_2_X = sparse.csr_matrix(result[10]) if polarity_2_X is None else sparse.vstack([polarity_2_X, result[10]])
        polarity_3_X = sparse.csr_matrix(result[11]) if polarity_3_X is None else sparse.vstack([polarity_3_X, result[11]])

        windows.append(result[12])

    f = open("results/best_back_window_scores.txt", "a")
    f.write(type + "\n")
    f.close()

    windows = np.array(windows)
    names = ["price_X", "volume_X", "price_all_X", "distribution_1_X", "distribution_2_X", "distribution_3_X", "sentiment_1_X", "sentiment_2_X", "sentiment_3_X", "polarity_1_X", "polarity_2_X", "polarity_3_X"]
    kmeans = KMeans(n_clusters=3, n_jobs=-1)

    for data, name in zip([price_X, volume_X, price_all_X, distribution_1_X, distribution_2_X, distribution_3_X, sentiment_1_X, sentiment_2_X, sentiment_3_X, polarity_1_X, polarity_2_X, polarity_3_X], names):
        labels = kmeans.fit_predict(data)

        window_sizes = []
        for label in Counter(labels).keys():
            _w = windows[np.where(labels == label, True, False)]
            window_sizes.append((np.mean(_w), np.std(_w)))

        window_sizes.sort(key=lambda x: x[0])
        window_size_stds = [std for mean, std in window_sizes]
        window_sizes = [mean for mean, std in window_sizes]
        f = open("results/best_back_window_scores.txt", "a")
        f.write("name: " + name + ", window_sizes: " + str(window_sizes) + ", window_size_stds: " + str(window_size_stds) + "\n")
        f.close()

    f = open("results/best_back_window_scores.txt", "a")
    f.write("\n")
    f.close()

