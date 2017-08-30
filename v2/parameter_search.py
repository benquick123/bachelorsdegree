import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from scipy.optimize import curve_fit
from scipy.misc import factorial
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture, GMM
from sklearn.feature_selection import RFECV
import pymongo
import time
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
import pickle

import matplotlib.pyplot as plt

import news as news
import trollbox as trollbox
import twitter as twitter
import common as common
import pickle_loading_saving as pls
import best_back_windows as bbw


def price_distribution(plot=True, **kwargs):
    print("PRICE DISTRIBUTION")
    # plot price distributions, choose threshold based on that distribution
    # ...so that majority class is as small as possible
    ids = np.array(kwargs["IDs"])
    type = kwargs["type"]
    window = kwargs["window"]
    final_set_f = kwargs["final_set_f"]
    del kwargs

    ids = final_set_f(None, None, ids)[0]

    price_changes = []
    if type == "articles":
        price_changes, _ = news.get_relative_Y_changes(ids, window)
    elif type == "conversations":
        price_changes, _ = trollbox.get_relative_Y_changes(ids, window)
    elif type == "tweets":
        price_changes, curr = twitter.get_relative_Y_changes(ids, window)
        from collections import Counter
        print(dict(Counter(curr)))
        print(len(curr))

    if price_changes is None or len(price_changes) == 0:
        exit()

    for i, price in enumerate(price_changes):
        price_changes[i] = abs(price)
    price_changes.sort()
    params = stats.alpha.fit(price_changes)

    if plot:
        plt.ioff()
        weights = np.ones_like(price_changes) / len(price_changes)

        n, _, _ = plt.hist(price_changes, bins=500, label="Porazdelitev cen", normed=True)

        # dist_names = ['alpha'] #, 'johnsonsu']

        """for dist_name in dist_names:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(price_changes)
                plt.plot(price_changes, dist.pdf(price_changes, *params), "-", label=dist_name)

                print(dist_name, np.sum(np.power(price_changes - dist.pdf(price_changes, *params), 2) / dist.pdf(price_changes, *params)))
            except TypeError:
                pass"""

        plt.plot(price_changes, stats.alpha.pdf(price_changes, *params), "-", label="Porazdelitev alpha")

        plt.title(str(window) + ", " + str(stats.alpha.ppf(1/3, *params)))
        plt.xlabel("relativna sprememeba cene")
        plt.legend()
        plt.xlim(0, 0.1)
        plt.yticks(np.linspace(0, max(n), 11), np.around(np.linspace(0, max(n), 11) / sum(n), 2))
        plt.savefig("figures/price_distribution_" + str(round(time.time()*1000)) + ".png")

    # this works only if price distribution is alpha:
    # threshold is determined so that sample data will be split in thirds.
    # https://i.stack.imgur.com/4o1Ex.png
    threshold1 = stats.alpha.ppf(1/3, *params)

    f = open("results/price_distribution.txt", "a")
    f.write(type + ", window: " + str(window) + ", thirds margin: " + str(threshold1) + "\n")
    f.close()

    return threshold1


def parallelized_matrix_creation(k, kwargs, window_range, back_window_range, raw_data, tfidf, vocabulary, ids, dates, data_X, get_Y_f, train_f, feature_selector, model):
    window = 300 * round((window_range[0] + np.random.rand() * (window_range[1] - window_range[0])) / 300)
    kwargs["window"] = window
    margin = price_distribution(plot=False, **kwargs)
    back_window_sizes = bbw.create_k_means(**kwargs)
    back_window_other = 300 * round((back_window_range[0] + np.random.rand() * (back_window_range[1] - back_window_range[0])) / 300)

    curr_tfidf_weight = 1
    _tfidf = None
    _topic_distributions = None
    _ns = None

    print(window, margin, back_window_sizes, back_window_other)
    for i, text in enumerate(raw_data):
        text["tfidf"] = tfidf[i]
        if text["_id"] in ids:
            if i % 100 == 0:
                print("iteration:", i)
            _, average_tfidf, _n, average_topics = common.get_averages_from_data(raw_data, text["date"], back_window_other, text["currency"], i, threshold=0.0, type="article")

            average_tfidf = text["tfidf"] * (curr_tfidf_weight / (_n + 1)) + (average_tfidf * (_n / (_n + 1))).multiply(text["tfidf"].power(0))
            _tfidf = sparse.csr_matrix(average_tfidf) if _tfidf is None else sparse.vstack([_tfidf, average_tfidf])

            average_topics = sparse.csr_matrix(list(text["topics"]) + list(average_topics))
            _topic_distributions = sparse.csr_matrix(average_topics) if _topic_distributions is None else sparse.vstack([_topic_distributions, average_topics])

            _ns = sparse.csr_matrix([1 / (_n+1)]) if _ns is None else sparse.vstack([_ns, sparse.csr_matrix([1 / (_n+1)])])

    labels = ["title_sentiment", "reduced_text_sentiment", "title_polarity", "reduced_text_polarity", "curr_in_title"]
    data_X = sparse.csr_matrix(data_X)

    labels = ["w"] + labels + vocabulary
    print(_ns.shape, data_X.shape, _tfidf.shape)
    data_X = sparse.hstack([_ns, data_X, _tfidf]).tocsr()

    labels = labels + ["topic_" + str(i) for i in range(len(raw_data[0]["topics"]))] + ["average_topic_" + str(i) for i in range(len(raw_data[0]["topics"]))]
    print(_topic_distributions.shape)
    data_X = sparse.hstack([data_X, _topic_distributions])

    l = np.array(pickle.load(open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + "articles" + "_labels_back_windows.pickle", "rb")))
    X = pickle.load(open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + "articles" + "_X_back_windows.pickle", "rb"))
    for back_window in back_window_sizes:
        indexes = np.zeros(len(l), dtype="bool")
        for i in range(len(indexes)):
            if int(l[i].split("_")[-1]) == back_window:
                indexes[i] = True

        history_attrs = X[:, indexes]
        labels = labels + l[indexes].tolist()
        print(history_attrs.shape)
        data_X = sparse.hstack([data_X, history_attrs])

    print(data_X.shape, len(labels))

    data_X = data_X.tocsr()
    data_Y = get_Y_f(ids, window, margin)
    _, score, precision, recall, _, classes = train_f(feature_selector, model, data_X, data_Y, "articles", dates, save=False, p=False, learn=True, test=False)
    result_string = "i: " + str(k) + ", score: " + str(score) + ", precision: " + str(precision) + ", recall: " + str(recall) + ", classes: " + str(classes) + "\n"
    result_string += "margin: " + str(margin) + ", window: " + str(window) + ", back_windows: " + str(back_window_sizes) + ", back_other: " + str(back_window_other) + "\n\n"
    
    f = open("/home/ubuntu/diploma/Proletarian 1.0/v2/results/parameter_search/data_parameters_" + "articles" + "_" + str(round(time.time()*1000)) + ".txt", "a")
    f.write(result_string)
    f.close()

    print("FILE SAVED")
    
    return data_X, labels, score, max(classes.values()) / sum(classes.values())


    """# general attr
        labels = ["w", "title_sentiment", "reduced_text_sentiment", "title_polarity", "reduced_text_polarity", "curr_in_title"]
        # averages
        labels += ["distribution_a_1200", "polarity_a_1200", "sentiment_a_1200"] + ["distribution_t_1200", "polarity_t_1200", "sentiment_t_1200"] + ["distribution_c_1200", "polarity_c_1200", "sentiment_c_1200"] + ["price_1200", "volume_1200", "price_all_1200"]
        labels += ["distribution_a_11100", "polarity_a_11100", "sentiment_a_11100"] + ["distribution_t_11100", "polarity_t_11100", "sentiment_t_11100"] + ["distribution_c_11100", "polarity_c_11100", "sentiment_c_11100"] + ["price_11100", "volume_11100", "price_all_11100"]
        labels += ["distribution_a_22800", "polarity_a_22800", "sentiment_a_22800"] + ["distribution_t_22800", "polarity_t_22800", "sentiment_t_22800"] + ["distribution_c_22800", "polarity_c_22800", "sentiment_c_22800"] + ["price_22800", "volume_22800", "volume_all_22800"]
        # tfidf
        labels += vocabulary
        # topics
        labels += ["topic_" + str(i) for i in range(len(raw_data[0]["topics"]))]
        labels += ["average_topic_" + str(i) for i in range(len(raw_data[0]["topics"]))]
        pls.save_labels(labels, type)"""


def parallelized_matrix_creation_old(k, window_range, margin_range, back_window_short_range, back_window_medium_range, back_window_long_range, back_window_range, type, ids, raw_data, data_X, train_f, get_dates_f, feature_selector, model, client, get_Y_f, date_key, currency_key, is_conversation, n_features, tfidf, kwargs, data_per_type, dates_per_type, articles, conversations, tweets):
    window = 300 * round((window_range[0] + np.random.rand() * (window_range[1] - window_range[0])) / 300)
    # window = 6600
    kwargs["window"] = window
    margin = price_distribution(plot=False, **kwargs)
    # margin = margin + margin_range[0] + np.random.rand() * (margin_range[1] - margin_range[0])
    back_window_short = 300 * round((back_window_short_range[0] + np.random.rand() * (back_window_short_range[1] - back_window_short_range[0])) / 300)
    back_window_medium = 300 * round((back_window_medium_range[0] + np.random.rand() * (back_window_medium_range[1] - back_window_medium_range[0])) / 300)
    back_window_long = 300 * round((back_window_long_range[0] + np.random.rand() * (back_window_long_range[1] - back_window_long_range[0])) / 300)
    back_window_other = 300 * round((back_window_range[0] + np.random.rand() * (back_window_range[1] - back_window_range[0])) / 300)

    # back_window_short = 1200
    # back_window_medium = 11100
    # back_window_long = 22800
    # back_window_other = 30000

    curr_tfidf_weight = 1
    _tfidf = None
    _topic_distributions = None
    _other_data = None

    back_windows = [back_window_short, back_window_medium, back_window_long]
    print(window, margin, back_windows, back_window_other)

    if not is_conversation:
        for i, text in enumerate(raw_data):
            text["tfidf"] = tfidf[i]
            if text["_id"] in ids:
                print("iteration", i)
                _, average_tfidf, _n, average_topics = common.get_averages_from_data(raw_data, text[date_key], back_window_other, text[currency_key], i, threshold=0.0, type=type[:-1])

                average_tfidf = text["tfidf"] * (curr_tfidf_weight / (_n + 1)) + (average_tfidf * (_n / (_n + 1))).multiply(text["tfidf"].power(0))
                _tfidf = sparse.csr_matrix(average_tfidf) if _tfidf is None else sparse.vstack([_tfidf, average_tfidf])
                average_topics = sparse.csr_matrix(list(text["topics"]) + list(average_topics))
                _topic_distributions = sparse.csr_matrix(average_topics) if _topic_distributions is None else sparse.vstack([_topic_distributions, average_topics])

                _other = []
                date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                for back_window in back_windows:
                    averages, _, _, _ = common.get_averages_from_data(raw_data, text[date_key], back_window, text[currency_key], i, threshold=0.0, type=type[:-1])
                    _other += averages
                    averages = common.get_averages_from_db(client, text[date_key], back_window, text[currency_key], articles=articles, tweets=tweets, conversations=conversations)
                    _other += averages

                    _other.append(common.get_price_change(client, text[currency_key], date_from - back_window, date_from))
                    total_curr_volume = common.get_total_volume(client, text[currency_key], date_from - back_window, date_from)
                    total_volume = common.get_total_volume(client, "all", date_from - back_window, date_from)
                    _other.append((total_curr_volume / total_volume) if total_volume > 0 else 0)
                    _other.append(common.get_all_price_changes(client, date_from - back_window, date_from))

                _other_data = sparse.csr_matrix(_other) if _other_data is None else sparse.vstack([_other_data, sparse.csr_matrix(_other)])
    else:
        for i, text in enumerate(raw_data):
            for currency in text[currency_key]:
                text["tfidf"] = tfidf[i]
                if str(text["_id"]) + ":" + currency in ids:
                    print("iteration", i)
                    _, average_tfidf, _n, _ = common.get_averages_from_data(raw_data, text[date_key], back_window_other, currency, i, threshold=0.0, type=type[:-1])

                    average_tfidf = text["tfidf"] * (curr_tfidf_weight / (_n + 1)) + (average_tfidf * (_n / (_n + 1))).multiply(text["tfidf"].power(0))
                    _tfidf = sparse.csr_matrix(average_tfidf) if _tfidf is None else sparse.vstack([_tfidf, average_tfidf])

                    _other = []
                    date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                    for back_window in back_windows:
                        averages, _, _, _ = common.get_averages_from_data(raw_data, text[date_key], back_window, currency, i, threshold=0.0, type=type[:-1])
                        _other += averages
                        averages = common.get_averages_from_db(client, text[date_key], back_window, currency, articles=articles, tweets=tweets, conversations=conversations)
                        _other += averages

                        _other.append(common.get_price_change(client, currency, date_from - back_window, date_from))
                        _other.append(common.get_total_volume(client, currency, date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from))
                        _other.append(common.get_all_price_changes(client, date_from - back_window, date_from))

                    _other_data = sparse.csr_matrix(_other) if _other_data is None else sparse.vstack([_other_data, sparse.csr_matrix(_other)])

    if data_X.shape[1] > n_features:
        data_X = data_X[:, :n_features]

    data_X = sparse.hstack([data_X, _other_data, _tfidf, _topic_distributions])
    if _topic_distributions is not None:
        data_X = sparse.hstack([data_X, _topic_distributions])
    data_X = data_X.tocsr()

    to_delete = np.zeros(data_X.shape[0], dtype="bool")
    for i in range(data_X.shape[0]):
        if not np.all(np.isfinite(data_X[i, :].todense())):
            to_delete[i] = True

    ids = np.array(ids)

    ids = ids[~to_delete]
    data_X = data_X[~to_delete, :]
    data_Y = get_Y_f(ids, window, margin)
    dates = get_dates_f(set(ids), raw_data, type)

    # pls.save_matrix_X(data_X, type)
    # pls.save_matrix_Y(data_Y, type, window, margin)
    # pls.save_matrix_IDs(ids, type)

    _, score, precision, recall, _, classes = train_f(feature_selector, model, data_X, data_Y, type, dates, save=False, p=False, learn=True, test=False)
    result_string = "i: " + str(k) + ", score: " + str(score) + ", precision: " + str(precision) + ", recall: " + str(recall) + ", classes: " + str(classes) + "\n"
    result_string += "margin: " + str(margin) + ", window: " + str(window) + ", back_windows: " + str(back_windows) + "back_other: " + str(back_window_other) + "\n\n"

    print(result_string)

    f = open("/home/ubuntu/diploma/Proletarian 1.0/v2/results/parameter_search/data_parameters_" + type + "_" + str(round(time.time()*1000)) + ".txt", "a")
    f.write(result_string)
    f.close()

    print("FILE SAVED")

    return result_string


def randomized_data_params_search(**kwargs):
    print("RANDOMIZED DATA PARAMS")
    n_iter = kwargs["n_iter"]
    back_window_range = kwargs["back_window_range"]
    window_range = kwargs["window_range"]
    data_X = kwargs["data_X"]
    labels = kwargs["labels"]
    train_f = kwargs["train_f"]
    get_dates_f = kwargs["dates_f"]
    feature_selector = kwargs["feature_selector"]
    model = kwargs["model"]

    is_conversation = False

    get_Y_f = news.get_Y
    tfidf_key = "reduced_text"
    raw_data = pls.load_data_pickle("articles")
    ids = np.array(pls.load_matrix_IDs("articles"))
    dates = get_dates_f(ids, raw_data, "articles")

    tfidf, vocabulary = common.calc_tf_idf(raw_data, 0.0, 1.0, tfidf_key, is_conversation)
    to_remove_mask = np.zeros(data_X.shape[1], dtype="bool")
    for i, label in enumerate(labels):
        if label == "w" or label.find(".") != -1 or label.find("polarity_") != -1 or label.find("sentiment_") != -1 or label.find("distribution_") != -1 or label.find("topic_") != -1 or label.find("price_") != -1 or label.find("volume_") != -1:
            to_remove_mask[i] = True

    data_X = data_X[:, ~to_remove_mask]

    n_iter = 1
    pool = ThreadPool(1)
    # k, kwargs, window_range, back_window_range, raw_data, tfidf, vocabulary, ids, dates, data_X, get_Y_f, train_f, feature_selector, model
    results = pool.starmap(parallelized_matrix_creation, zip(list(range(n_iter)), repeat(kwargs), repeat(window_range), repeat(back_window_range), repeat(raw_data), repeat(tfidf), repeat(vocabulary), repeat(ids), repeat(dates), repeat(data_X), repeat(get_Y_f), repeat(train_f), repeat(feature_selector), repeat(model)))
    pool.close()
    pool.join()
    
    final_X, final_labels = None, []
    best_score = 0
    for result in results:
        print(result[2], result[3])
        if result[2] - result[3] > best_score:
            best_score = result[2] - result[3]
            final_X = result[0]
            final_labels = result[1]
            
    # pls.save_matrix_X(final_X, "articles")
    # pls.save_labels(final_labels, "articles")
    print(best_score)


    # f = open("results/data_params_search_results.txt", "a")
    # for result in results:
    #     f.write(result)
    # f.close()

