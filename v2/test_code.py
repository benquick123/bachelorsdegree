import matplotlib.pyplot as plt

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

import news as news
import trollbox as trollbox
import twitter as twitter
import common as common
import pickle_loading_saving as pls


def optimal_attr_number(plot=True, **kwargs):
    print("FIND OPTIMAL ARRTIBUTE NUMBER")
    # optimal attribute number
    # learn model several times with changing threshold.
    n_iter = kwargs["n_iter"]
    threshold_range = kwargs["threshold_range"]
    train_f = kwargs["train_f"]
    n = kwargs["n"]
    feature_selector = kwargs["feature_selector"]
    model = kwargs["model"]
    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    type = kwargs["type"]
    del kwargs

    i = 0
    best_threshold = 0
    best_score = 0
    scores = []
    f = open("results/attribute_selection.txt", "a")
    f.write(type + ", " + str(feature_selector)[:10] + "\n")

    while i < n_iter:
        threshold = threshold_range[0] + np.random.rand() * (threshold_range[1] - threshold_range[0])

        # threshold = threshold_range[0] + (i / n_iter) * (threshold_range[1] - threshold_range[0])
        _, score, score_std, precision, recall, matrix_shape, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
        if score >= best_score:
            best_threshold = threshold
            best_score = score

        f.write(str(i) + " - threshold: " + str(threshold) + ", score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", ~n_attr: " + str(matrix_shape[1]) + "\n")
        scores.append((score, score_std, matrix_shape[1]))
        i += 1

    f.write("\n")
    f.close()

    #plot
    if plot:
        scores.sort(key=lambda x: x[2])
        plt.plot([shape for _, _, shape in scores], [sc for sc, _, _ in scores])
        plt.fill_between([shape for _, _, shape in scores], np.array([sc for sc, _, _ in scores]) - np.array([std for _, std, _ in scores]), np.array([sc for sc, _, _ in scores]) + np.array([std for _, std, _ in scores]), alpha=0.5)
        # plt.errorbar([shape for _, _, shape in scores], [sc for sc, _, _ in scores], yerr=[std for _, std, _ in scores])
        plt.show()

    print(best_score, best_threshold)
    return best_score, best_threshold


# checked
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
    params = stats.lognorm.fit(price_changes, 2)
    # exit()

    # res = Poisson(price_changes, np.ones_like(price_changes)).fit()
    # print(res.summary())
    # mean = res.predict()
    # print(mean)
    # dist = stats.poisson(mean[0])

    if plot:
        weights = np.ones_like(price_changes) / len(price_changes)

        plt.hist(price_changes, bins=1000, label="Porazdelitev cen", normed=True)
        plt.plot(price_changes, stats.lognorm.pdf(price_changes, *params), "-", label="Logaritemsko normalna porazdelitev")

        plt.xlabel("relativna spremmeba cene")
        plt.legend()
        plt.show()

    # this works only if price distribution is cauchy:
    # threshold is determined so that sample data will be split in thirds.
    # https://i.stack.imgur.com/4o1Ex.png
    threshold1 = stats.lognorm.ppf(1/3, *params)

    f = open("results/price_distribution.txt", "a")
    f.write(type + ", window: " + str(window) + ", thirds margin: " + str(threshold1) + "\n")
    f.close()

    print(threshold1)

    return threshold1


# checked
def optimal_margin(plot=True, **kwargs):
    print("FIND OPTIMAL MARGIN")
    # also: how does majority class size affect classification accuracy - plot
    n_iter = kwargs["n_iter"]
    margin_range = kwargs["margin_range"]
    train_f = kwargs["train_f"]
    n = kwargs["n"]
    feature_selector = kwargs["feature_selector"]
    model = kwargs["model"]
    data_X = kwargs["data_X"]
    threshold = kwargs["threshold"]
    raw_data = kwargs["raw_data"]
    ids = kwargs["IDs"]
    type = kwargs["type"]
    window = kwargs["window"]
    del kwargs

    func = None
    if type == "articles":
        news.articles = raw_data
        func = news.get_Y
    elif type == "conversations":
        trollbox.conversations = raw_data
        func = trollbox.get_Y
    elif type == "tweets":
        twitter.tweets = raw_data
        func = twitter.get_Y

    i = 0
    best_score = 0
    best_margin = 0
    scores = []
    f = open("results/majority_class_scores.txt", "a")
    f.write(type + ", " + str(model)[:10] + "\n")

    while i < n_iter:
        # margin = margin_range[0] + np.random.rand() * (margin_range[1] - margin_range[0])
        margin = margin_range[0] + ((i + 1) / (n_iter + 1)) * (margin_range[1] - margin_range[0])
        data_Y = func(ids, window, margin)
        _, score, score_std, precision, recall, _, classes = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)

        if score - max(classes.values()) / sum(classes.values()) > best_score:
            best_score = score - max(classes.values()) / sum(classes.values())
            best_margin = margin

        f.write(str(i) + " - margin: " + str(margin) + ", score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", classes: " + str(classes) + "\n")
        scores.append((score - max(classes.values()) / sum(classes.values()), score_std, max(classes.values()) / sum(classes.values())))
        i += 1

    f.write("\n")
    f.close()
    if plot:
        scores.sort(key=lambda x: x[2])
        plt.plot([major for _, _, major in scores], [sc for sc, _, _ in scores], "-")
        plt.fill_between([major for _, _, major in scores], np.array([sc for sc, _, _ in scores]) - np.array([std for _, std, _ in scores]), np.array([sc for sc, _, _ in scores] + np.array([std for _, std, _ in scores])), alpha=0.5, label="Klasifikacijska točnost s standardnim odklonom")
        plt.axhline(y=0)
        plt.ylabel("klasifikacijska točnost - večinski razred")
        plt.xlabel("večinski razred")
        plt.legend()
        plt.show()

    print(best_score, best_margin)
    return best_score, best_margin


def optimal_window(**kwargs):
    print("FIND OPTIMAL WINDOW")
    # plot classification accuracy vs. prediction window size
    n_iter = kwargs["n_iter"]
    window_range = kwargs["window_range"]
    train_f = kwargs["train_f"]
    n = kwargs["n"]
    feature_selector = kwargs["feature_selector"]
    model = kwargs["model"]
    data_X = kwargs["data_X"]
    raw_data = kwargs["raw_data"]
    threshold = kwargs["threshold"]
    ids = kwargs["IDs"]
    margin = kwargs["margin"]
    type = kwargs["type"]

    func = None
    if type == "articles":
        news.articles = raw_data
        func = news.get_Y
    elif type == "conversations":
        trollbox.conversations = raw_data
        func = trollbox.get_Y
    elif type == "tweets":
        twitter.tweets = raw_data
        func = twitter.get_Y

    i = 0
    best_score = 0
    best_window = 0
    scores = []
    f = open("results/predict_window_scores.txt", "a")
    f.write(type + ", model: " + str(model)[:10] + "\n")

    while i < n_iter:
        window = 300 * round((window_range[0] + np.random.rand() * (window_range[1] - window_range[0])) / 300)
        kwargs["window"] = window
        margin = price_distribution(**kwargs)
        data_Y = func(ids, window, margin)
        _, score, score_std, _, _, _, classes = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)

        if score - max(classes.values()) / sum(classes.values()) > best_score:
            best_score = score - max(classes.values()) / sum(classes.values())
            best_window = window

        f.write(str(i) + " - window: " + str(window) + ", margin: " + str(margin) + ", score: " + str(score) + " (+/- " + str(score_std) + "), classes: " + str(classes) + "\n")
        scores.append((score, score_std, window, max(classes.values()) / sum(classes.values())))
        i += 1

    f.write("\n")
    f.close()

    scores.sort(key=lambda x: x[2])
    plt.plot([w for _, _, w, _ in scores], [sc - major for sc, _, _, major in scores])
    plt.fill_between([w for _, _, w, _ in scores], np.array([sc - major for sc, _, _, major in scores]) - np.array([std for _, std, _, _ in scores]), np.array([sc - major for sc, _, _, major in scores]) + np.array([std for _, std, _, _ in scores]), alpha=0.5)
    # plt.errorbar([w for _, _, w, _ in scores], [sc for sc, _, _, _ in scores], yerr=[std for _, std, _, _ in scores])
    plt.plot([w for _, _, w, _ in scores], [major for _, _, _, major in scores], "-o")
    plt.show()

    print(best_score, best_window)
    return best_score, best_window


def optimal_back_window(plot=True, **kwargs):
    print("OPTIMAL BACK_WINDOW")
    # what are best values for back_window sizes?
    # build X matrix multiple times and find best sizes using regression
    n_iter = kwargs["n_iter"]
    type = kwargs["type"]
    raw_data = kwargs["raw_data"]
    window_range = kwargs["window_range"]
    margin = kwargs["margin"]
    back_window_range = kwargs["back_window_range"]
    back_window_ratio = kwargs["back_window_ratio"]
    ids = np.array(kwargs["IDs"])
    final_set_f = kwargs["final_set_f"]
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    del kwargs

    ids = set(final_set_f(None, None, IDs=ids)[0])

    is_conversation = False
    date_key = ""
    currency_key = ""
    articles = True
    conversations = True
    tweets = True
    # _data_Y = []
    price_changes_f = None
    if type == "articles":
        date_key = "date"
        currency_key = "currency"
        articles = False
        # _data_Y = news.get_relative_Y_changes(ids, window)[0]
        price_changes_f = news.get_relative_Y_changes
    elif type == "conversations":
        is_conversation = True
        date_key = "conversation_end"
        currency_key = "coin_mentions"
        conversations = False
        # _data_Y = trollbox.get_relative_Y_changes(ids, window)[0]
        price_changes_f = trollbox.get_relative_Y_changes
    elif type == "tweets":
        date_key = "posted_time"
        currency_key = "crypto_currency"
        tweets = False
        # _data_Y = twitter.get_relative_Y_changes(ids, window)[0]
        price_changes_f = twitter.get_relative_Y_changes

    i = 0
    best_score = 0
    best_back_window = 0
    scores = []
    f = open("results/predict_back_window_scores_" + type + ".txt", "a")
    f.write(type + "\n")
    f.close()
    while i < n_iter:
        # back_window = 300 * round((back_window_range[0] + np.random.rand() * (back_window_range[1] - back_window_range[0])) / 300)
        window = 300 * round((window_range[0] + np.random.rand() * (window_range[1] - window_range[0])) / 300)
        back_window = 300 * round(window * (back_window_ratio[0] + np.random.rand() * (back_window_ratio[1] - back_window_ratio[0])) / 300)
        _data_Y, _ = price_changes_f(ids, window)
        data_X = []
        data_Y = []
        k = 0
        if not is_conversation:
            for j, text in enumerate(raw_data):
                if text["_id"] in ids:
                    print("j: ", j, ", id: ", text["_id"], sep="")
                    date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                    db_averages = common.get_averages_from_db(client, text[date_key], window, text[currency_key], tweets=tweets, articles=articles, conversations=conversations)
                    data_averages = common.get_averages_from_data(raw_data, text[date_key], window, text[currency_key], j, threshold=0.0, type=type[:-1], data_averages_only=True)

                    price = common.get_price_change(client, text[currency_key], date_from - back_window, date_from)
                    volume = common.get_total_volume(client, text[currency_key], date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from)
                    price_all = common.get_all_price_changes(client, date_from - back_window, date_from)

                    data = np.array(db_averages + data_averages + [price, volume, price_all])
                    if np.all(np.isfinite(data)):
                        data_X.append(db_averages + data_averages + [price, volume, price_all])
                        data_Y.append(_data_Y[k])
                    k += 1

        else:
            for j, text in enumerate(raw_data):
                for currency in text[currency_key]:
                    if str(text["_id"]) + ":" + currency in ids:
                        print("j: ", j, ", id: ", str(text["_id"]) + ":" + currency, sep="")
                        date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
                        db_averages = common.get_averages_from_db(client, text[date_key], window, currency, tweets=tweets, articles=articles, conversations=conversations)
                        data_averages = common.get_averages_from_data(raw_data, text[date_key], window, currency, j, threshold=0.0, type=type[:-1], data_averages_only=True)

                        price = common.get_price_change(client, currency, date_from - back_window, date_from)
                        volume = common.get_total_volume(client, currency, date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from)
                        price_all = common.get_all_price_changes(client, date_from - back_window, date_from)

                        data = db_averages + data_averages + [price, volume, price_all]
                        if np.all(np.isfinite(data)):
                            data_X.append(data)
                            data_Y.append(_data_Y[k])
                        k += 1

        model = LinearRegression(copy_X=False)
        data_X = sparse.csr_matrix(data_X)
        indexes = np.array(np.linspace(0, data_X.shape[0], 4), dtype="int")
        scores = []
        for k in range(len(indexes)-1):
            model.fit(data_X[indexes[k]:indexes[k+1], :], data_Y[indexes[k]:indexes[k+1]])
            scores.append(model.score(data_X[indexes[k]:indexes[k+1], :], data_Y[indexes[k]:indexes[k+1]]))

        score = np.mean(scores)
        print("SCORE:", score)
        scores.append((score, back_window))
        if best_score > score:
            best_score = score
            best_back_window = back_window

        f = open("results/predict_back_window_scores_" + type + ".txt", "a")
        f.write(str(i) + " - r^2: " + str(score) + ", back_window: " + str(back_window) + ", window: " + str(window) + ", margin: " + str(margin) + "\n")
        f.close()

        i += 1

    f = open("results/predict_back_window_scores_" + type + ".txt", "a")
    f.write("\n")
    f.close()

    if plot:
        scores.sort(key=lambda x: x[1])
        plt.plot([w for _, w in scores], [sc for sc, _ in scores], "-")
        plt.show()

    print(best_score, best_back_window)
    return best_score, best_back_window


# ----------------------------------------------------------------------------------------------------

# checked
def tfidf_test(**kwargs):
    print("TEST TF-IDF")
    # TF-IDF - build several X matrices:
    #   - with fully weighted TF-IDF
    #   - with non-weighted TF-IDF
    #   - weighted just with sentiments
    #   - weighted just with financial words weights
    #       -> which words in financial dict are in the final (reduced) attribute set?
    #   - weighted with sentiment + financial words weights
    #   - weighted just with previous TF-IDFs

    raw_data = kwargs["raw_data"]
    type = kwargs["type"]
    train_f = kwargs["train_f"]
    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    n = kwargs["n"]
    model = kwargs["model"]
    feature_selector = kwargs["feature_selector"]
    ids = set(kwargs["IDs"])
    labels = kwargs["labels"]
    margin = kwargs["margin"]
    window = kwargs["window"]
    del kwargs

    dict_key = ""
    date_key = ""
    currency_key = ""
    is_conversation = False
    if type == "articles":
        dict_key = "reduced_text"
        date_key = "date"
        currency_key = "currency"
    elif type == "conversations":
        dict_key = "clean_text"
        date_key = "conversation_end"
        currency_key = "coin_mentions"
        is_conversation = True
    elif type == "tweets":
        dict_key = "clean_text"
        date_key = "posted_time"
        currency_key = "crypto_currency"

    tfidf_indexes = np.zeros(len(labels), dtype="bool")
    for i, label in enumerate(labels):
        if label.find(".") != -1:
            tfidf_indexes[i] = True

    tfidf, vocabulary = common.calc_tf_idf(raw_data, min_df=0.0, max_df=1.0, dict_key=dict_key, is_conversation=is_conversation)
    remove_tfidf_lines = np.zeros(tfidf.shape[0], dtype="bool")
    if not is_conversation:
        for i, text in enumerate(raw_data):
            if text["_id"] not in ids:
                remove_tfidf_lines[i] = True
        tfidf = tfidf[~remove_tfidf_lines, :]
    else:
        _tfidf = None
        for i, text in enumerate(raw_data):
            raw_data[i]["tfidf"] = tfidf[i, :]
            for currency in text["coin_mentions"]:
                if str(text["_id"]) + ":" + currency in ids:
                    if _tfidf is None:
                        _tfidf = sparse.csr_matrix(tfidf[i, :])
                    else:
                        _tfidf = sparse.vstack([_tfidf, tfidf[i, :]])
        tfidf = sparse.csr_matrix(_tfidf)

    max_n = 50000
    if data_X.shape[0] > max_n:
        data_X = data_X[:max_n, :]
        data_Y = data_Y[:max_n]
        tfidf = tfidf[:max_n, :]

    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write(type + "\n")
    f.write("non weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    financial_weights = common.create_financial_vocabulary(vocabulary) + 1
    _f = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _f.setdiag(financial_weights)
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _f]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("only financial weights - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    sentiment_weights = common.calc_sentiment_for_vocabulary(vocabulary)
    sentiment_weights = np.where(sentiment_weights >= 0, sentiment_weights + 1, sentiment_weights - 1)
    _s = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _s.setdiag(sentiment_weights)
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _s]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("only sentiment weights - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    weights = sparse.csr_matrix(financial_weights * sentiment_weights)
    _w = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _w.setdiag(weights.toarray()[0])
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _w]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("sentiment & financial weights - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    back_window = 3600 * 3
    # if previous TF-IDFs contribute to accuracy, try different weights
    curr_tfidf_weight = 1
    k = 0

    if not is_conversation:
        for i, text in enumerate(raw_data):
            # if k >= max_n:
            #     break
            if text["_id"] in ids:
                text["tfidf"] = tfidf[k]
                _, average_tfidf, _n, _ = common.get_averages_from_data(raw_data, text[date_key], back_window, text[currency_key], i, threshold=0.0, type=type[:-1])
                tfidf[k, :] = tfidf[k, :] * (curr_tfidf_weight / (_n+1)) + (average_tfidf * (_n / (_n+1))).multiply(tfidf[k, :].power(0))
                k += 1
    else:
        for i, text in enumerate(raw_data):
            for currency in text[currency_key]:
                # if k >= max_n:
                #     break
                if str(text["_id"]) + ":" + currency in ids:
                    _, average_tfidf, _n, _ = common.get_averages_from_data(raw_data, text[date_key], back_window, currency, i, threshold=0.0, type=type[:-1])
                    tfidf[k, :] = tfidf[k, :] * (curr_tfidf_weight / (_n + 1)) + (average_tfidf * (_n / (_n + 1))).multiply(tfidf[k, :].power(0))
                    k += 1

    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("average_tfidf weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + ", back_window: " + str(back_window) + ", curr_tfidf_weight: " + str(curr_tfidf_weight) + "\n")
    f.close()

    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _s]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("average_tfidf and sentiment weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + ", back_window: " + str(back_window) + ", curr_tfidf_weight: " + str(curr_tfidf_weight) + "\n")
    f.close()

    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _f]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("average_tfidf and financial weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + ", back_window: " + str(back_window) + ", curr_tfidf_weight: " + str(curr_tfidf_weight) + "\n")
    f.close()

    financial_weights = common.create_financial_vocabulary(vocabulary) + 1
    sentiment_weights = common.calc_sentiment_for_vocabulary(vocabulary)
    sentiment_weights = np.where(sentiment_weights >= 0, sentiment_weights + 1, sentiment_weights - 1)
    weights = sparse.csr_matrix(financial_weights * sentiment_weights)
    _w = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _w.setdiag(weights.toarray()[0])
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _w]).tocsr()
    pls.save_matrix_X(data_X, "conversations", k=20000)
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("average_tfidf, sentiment and financial weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + ", back_window: " + str(back_window) + ", curr_tfidf_weight: " + str(curr_tfidf_weight) + "\n")
    f.close()

    data_X = data_X[:, ~tfidf_indexes]
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("without tfidf - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.write("\n")
    f.close()


# checked
def topics_test(**kwargs):
    print("TEST TOPICS")
    # topics - build X matrices:
    #   - without topics
    #   - with current topics only
    #   - with past topics only
    #   - with both sets of features

    type = kwargs["type"]
    train_f = kwargs["train_f"]
    final_set_f = kwargs["final_set_f"]
    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    raw_data = kwargs["raw_data"]
    ids = np.array(kwargs["IDs"])
    n = kwargs["n"]
    model = kwargs["model"]
    feature_selector = kwargs["feature_selector"]
    labels = kwargs["labels"]
    margin = kwargs["margin"]
    window = kwargs["window"]
    del kwargs

    if type == "conversations":
        print("conversations have no topics!")
        exit()

    topic_indexes = np.zeros(len(labels), dtype="bool")
    average_topic_indexes = np.zeros(len(labels), dtype="bool")
    for i, label in enumerate(labels):
        if label.find("average_topic") != -1:
            average_topic_indexes[i] = True
        elif label.find("topic") != -1:
            topic_indexes[i] = True             # topic indexes are the same as average topic indexes; consequence of deletion from matrix

    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("full - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    average_topics = data_X[:, average_topic_indexes]
    data_X = data_X[:, ~average_topic_indexes]
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("topics only - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    data_X = data_X[:, ~topic_indexes[:data_X.shape[1]]]
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("none: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    data_X = sparse.hstack([data_X, average_topics]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("average topics only - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    date_key = ""
    currency_key = ""
    if type == "articles":
        date_key = "date"
        currency_key = "currency"
    elif type == "tweets":
        date_key = "posted_time"
        currency_key = "crypto_currency"

    back_window = 6 * 3600
    average_topics = []
    for i, text in enumerate(raw_data):
        if text["_id"] in ids:
            _, _, _, _average_topics = common.get_averages_from_data(raw_data, text[date_key], back_window, text[currency_key], i, threshold=0.0, type=type[:-1])
            average_topics.append(_average_topics)

    average_topics = np.array(average_topics)
    average_topics = sparse.csr_matrix(average_topics)
    data_X = data_X[:, :-average_topics.shape[1]]
    data_X = sparse.hstack([data_X, average_topics]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("average topics only - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + ", back_window: " + str(back_window) + "\n")
    f.write("\n")
    f.close()


def technical_test(**kwargs):
    print("TECHNICAL TEST")
    # tabulate contribution coefficients for technical data.
    # also, build X matrices:
    #   - with past prices only
    #   - with past volumes only
    #   - with overall market movements only
    #   - without any of them
    #   - with all of them

    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    labels = kwargs["labels"]
    train_f = kwargs["train_f"]
    n = kwargs["n"]
    feature_selector = kwargs["feature_selector"]
    model = kwargs["model"]
    threshold = kwargs["threshold"]
    type = kwargs["type"]
    window = kwargs["window"]
    margin = kwargs["margin"]
    del kwargs

    price_all_labels = np.zeros(len(labels), dtype="bool")
    price_labels = np.zeros(len(labels), dtype="bool")
    volume_labels = np.zeros(len(labels), dtype="bool")
    for i, label in enumerate(labels):
        if label.find("price_all") != -1:
            price_all_labels[i] = True
        elif label.find("price") != -1:
            price_labels[i] = True
        elif label.find("volume") != -1:
            volume_labels[i] = True

    all_prices = data_X[:, price_all_labels]
    prices = data_X[:, price_labels]
    volumes = data_X[:, volume_labels]

    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/technical_test_scores", "a")
    f.write(type + " - window: " + str(window) + ", margin: " + str(margin) + "\n")
    f.write("past prices - scores: " + str(score) + ", scores_std: " + str(score_std) + ", precision: " + str(precision) + ", recall: " + str(recall) + "\n")
    f.close()

    data_X = data_X[:, price_labels | price_all_labels | volume_labels]
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/technical_test_scores", "a")
    f.write("no technical - scores: " + str(score) + ", scores_std: " + str(score_std) + ", precision: " + str(precision) + ", recall: " + str(recall) + "\n")
    f.close()

    data_X = sparse.hstack([data_X, prices]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/technical_test_scores", "a")
    f.write("past prices - scores: " + str(score) + ", scores_std: " + str(score_std) + ", precision: " + str(precision) + ", recall: " + str(recall) + "\n")
    f.close()

    data_X = data_X[:, :-prices.shape[1]]
    data_X = sparse.hstack([data_X, all_prices])
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/technical_test_scores", "a")
    f.write("all past prices - scores: " + str(score) + ", scores_std: " + str(score_std) + ", precision: " + str(precision) + ", recall: " + str(recall) + "\n")
    f.close()

    data_X = data_X[:, :-all_prices.shape[1]]
    data_X = sparse.hstack([data_X, volumes])
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/technical_test_scores", "a")
    f.write("past prices - scores: " + str(score) + ", scores_std: " + str(score_std) + ", precision: " + str(precision) + ", recall: " + str(recall) + "\n")
    f.write("\n")
    f.close()


# -----------------------------------------------------------------------------------------------------

def get_most_info_windows(**kwargs):
    n_iter = kwargs["n_iter"]
    type = kwargs["type"]
    raw_data = kwargs["raw_data"]
    window_range = kwargs["window_range"]
    margin = kwargs["margin"]
    back_window_range = kwargs["back_window_range"]
    back_window_ratio = kwargs["back_window_ratio"]
    ids = np.array(kwargs["IDs"])
    final_set_f = kwargs["final_set_f"]
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    del kwargs

    if type == "articles":
        pass
    elif type == "conversations":
        pass
    elif type == "tweets":
        pass

    i = 0
    while i < n_iter:
        pass


def randomized_data_params_search(**kwargs):
    print("RANDOMIZED DATA PARAMS")
    n_iter = kwargs["n_iter"]
    back_window_short_range = kwargs["back_window_short"]
    back_window_long_range = kwargs["back_window_long"]
    back_window_medium_range = kwargs["back_window_medium"]
    back_window_range = kwargs["back_window_range"]
    window_range = kwargs["window_range"]
    margin_range = kwargs["margin_range"]
    type = kwargs["type"]
    ids = np.array(kwargs["IDs"])
    raw_data = kwargs["raw_data"]
    data_X = kwargs["data_X"]
    labels = kwargs["labels"]
    train_f = kwargs["train_f"]
    get_dates_f = kwargs["dates_f"]
    feature_selector = kwargs["feature_selector"]
    model = kwargs["model"]
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)

    get_Y_relative_f = None
    get_Y_f = None
    tfidf_key = ""
    date_key = ""
    currency_key = ""
    is_conversation = False
    articles = conversations = tweets = True

    if type == "articles":
        get_Y_relative_f = news.get_relative_Y_changes
        get_Y_f = news.get_Y
        tfidf_key = "reduced_text"
        date_key = "date"
        currency_key = "currency"
        articles = False
    elif type == "conversations":
        get_Y_relative_f = trollbox.get_relative_Y_changes
        get_Y_f = trollbox.get_Y
        tfidf_key = "clean_text"
        date_key = "conversation_end"
        currency_key = "coin_mentions"
        is_conversation = True
        conversations = False
    elif type == "tweets":
        get_Y_relative_f = twitter.get_relative_Y_changes
        get_Y_f = twitter.get_Y
        tfidf_key = "clean_text"
        date_key = "posted_time"
        currency_key = "crypto_currency"
        tweets = False

    tfidf, vocabulary = common.calc_tf_idf(raw_data, 0.0, 1.0, tfidf_key, is_conversation)
    to_remove_mask = np.zeros(data_X.shape[1], dtype="bool")
    for i, label in enumerate(labels):
        if label.find(".") != -1 or label.find("polarity_") != -1 or label.find("sentiment_") != -1 or label.find("distribution_") != -1 or label.find("topic_") != -1 or label.find("price_") != -1 or label.find("volume_") != -1:
            to_remove_mask[i] = True

    data_X = data_X[:, ~to_remove_mask]
    n_features = data_X.shape[1]

    i = 0
    while i < n_iter:
        window = 300 * round((window_range[0] + np.random.rand() * (window_range[1] - window_range[0])) / 300)
        margin = price_distribution(plot=False, **kwargs)
        margin = margin + margin_range[0] + np.random.rand() * (margin_range[1] - margin_range[0])
        back_window_short = 300 * round((back_window_short_range[0] + np.random.rand() * (back_window_short_range[1] - back_window_short_range[0])) / 300)
        back_window_medium = 300 * round((back_window_medium_range[0] + np.random.rand() * (back_window_medium_range[1] - back_window_medium_range[0])) / 300)
        back_window_long = 300 * round((back_window_long_range[0] + np.random.rand() * (back_window_long_range[1] - back_window_long_range[0])) / 300)
        back_window_other = 300 * round((back_window_range[0] + np.random.rand() * (back_window_range[1] - back_window_range[0])) / 300)

        curr_tfidf_weight = 1
        _tfidf = None
        _topic_distributions = None
        _other_data = None

        back_windows = [back_window_short, back_window_medium, back_window_long]

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
                        _other.append(common.get_total_volume(client, text[currency_key], date_from - back_window, date_from) / common.get_total_volume(client, "all", date_from - back_window, date_from))
                        _other.append(common.get_all_price_changes(client, date_from - back_window, date_from))

                    _other_data = sparse.csr_matrix(_other) if _other_data is None else sparse.vstack([_other_data, sparse.csr_matrix(_other)])
        else:
            for i, text in enumerate(raw_data):
                for currency in text[currency_key]:
                    text["tfidf"] = tfidf[i]
                    if str(text["_id"]) + ":" + currency in ids:
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

        data_X = sparse.hstack([data_X, _other, _tfidf])
        if _topic_distributions is not None:
            data_X = sparse.hstack([data_X, _topic_distributions])
        data_X = data_X.tocsr()

        data_Y = get_Y_f(ids, window, margin)

        dates = get_dates_f(set(ids), raw_data, type)

        train_f(feature_selector, model, data_X, data_Y, dates, save=False, p=False, learn=True, test=False)



def randomized_model_params_search(**kwargs):
    print("GRID SEARCH")
    # do grid search to find best params for every learning algorithm used

    n = kwargs["n"]
    n_iter = kwargs["n_iter"]
    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    model = kwargs["model"]
    type = kwargs["type"]
    threshold = kwargs["threshold"]
    feature_selector = kwargs["feature_selector"]
    final_set_f = kwargs["final_set_f"]
    del kwargs

    data_X, data_Y, _, _ = final_set_f(data_X, np.array(data_Y))
    sfm = SelectFromModel(feature_selector, threshold=str(threshold) + "*mean")
    sfm.fit(data_X, data_Y)
    data_X = sfm.transform(data_X)

    params = dict()
    if isinstance(model, LinearSVC):
        params = {"C": stats.randint(1, 10), "penalty": ["l1", "l2"], "dual": [True, False], "loss": ["hinge", "squared_hinge"]}
    elif isinstance(model, MLPClassifier()):
        params = {"solver": ["lbfgs", "sgd", "adam"], "hidden_layer_sizes": tuple([stats.randint(50, 200)])}
    elif isinstance(model, RandomForestClassifier()):
        params = {"n_estimators": stats.randint(5, 40), "max_features": [None, "auto", stats.randint(1000, data_X.shape[1])], "min_samples_leaf": stats.randint(1, 100)}
    elif isinstance(model, KNeighborsClassifier()):
        params = {"k_neighbours": stats.randint(2, 20), "algorithm": ["auto", "ball_tree", "kd_tree"], "leaf_size": stats.randint(10, 40), "p": [1, 2]}
    else:
        print("unknown classifier!")
        exit()

    rs = RandomizedSearchCV(model, param_distributions=params, cv=n, n_iter=n_iter, error_score=0, verbose=1000)
    rs.fit(data_X, data_Y)
    results = rs.cv_results_

    f = open("results/params_search_results.txt", "a")
    f.write(type + ", model: " + str(model)[:10] + "\n")
    for key in results.keys():
        f.write(key + "\t")
    f.write("\n")
    for i in range(n_iter):
        for key, item in results.items():
            f.write(str(item[i]) + "\t")
        f.write("\n")
    f.write("\n")
    f.close()

    print(rs.best_score_, rs.best_params_)
    return rs.best_score_, rs.best_params_


def feature_selection_test(**kwargs):
    print("FEATURE SELECTION TEST")
    # how do different feature selection algorithms affect classification accuracy
    feature_selectors = kwargs["feature_selectors"]
    train_f = kwargs["train_f"]
    n = kwargs["n"]
    model = kwargs["model"]
    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    type = kwargs["type"]

    for i, feature_selector in enumerate(feature_selectors):
        kwargs["feature_selector"] = feature_selector
        _, threshold = optimal_attr_number(plot=False, **kwargs)
        _, score, score_std, precision, recall, matrix_shape, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)

        f = open("results/feature_selection.txt", "a")
        if i == 0:
            f.write(type + ", threshold: " + str(threshold) + "\n")
        f.write(str(feature_selector)[:10] + " - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + "~n_attr: " + matrix_shape[1] + "\n")
        if i == len(feature_selectors) - 1:
            f.write("\n")
        f.close()


