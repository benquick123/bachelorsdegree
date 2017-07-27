import matplotlib.pyplot as plt
import seaborn
import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
from statsmodels.discrete.discrete_model import Poisson

import v2.news as news
import v2.trollbox as trollbox
import v2.twitter as twitter
import v2.common as common


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


def price_distribution(plot=True, **kwargs):
    print("PRICE DISTRIBUTION")
    # plot price distributions, choose threshold based on that distribution
    # ...so that majority class is as small as possible
    ids = kwargs["IDs"]
    type = kwargs["type"]
    window = kwargs["window"]
    del kwargs

    price_changes = []
    if type == "articles":
        price_changes, _ = news.get_relative_Y_changes(ids, window)
    elif type == "conversations":
        price_changes, _ = trollbox.get_relative_Y_changes(ids, window)
    elif type == "tweets":
        price_changes, _ = twitter.get_relative_Y_changes(ids, window)

    if price_changes is None or len(price_changes) == 0:
        exit()

    # for i in range(len(price_changes)):
    #     if price_changes[i] < 0:
    #         price_changes[i] = abs(price_changes[i])

    price_changes.sort()
    mean, deviation = stats.cauchy.fit(price_changes)

    # res = Poisson(price_changes, np.ones_like(price_changes)).fit()
    # print(res.summary())
    # mean = res.predict()
    # print(mean)
    # dist = stats.poisson(mean[0])

    if plot:
        plt.hist(price_changes, normed=True, bins=1000)
        plt.plot(price_changes, stats.cauchy.pdf(price_changes, mean, deviation), "-")
        # plt.plot(price_changes, dist.pmf(price_changes))
        plt.draw()
        plt.show()

    # this works only if price distribution is cauchy:
    # threshold is determined so that sample data will be split in thirds.
    # https://i.stack.imgur.com/4o1Ex.png
    threshold1 = abs(stats.cauchy.ppf(2/3, mean, deviation))
    threshold2 = abs(stats.cauchy.ppf(1/3, mean, deviation))
    thirds_split = (threshold1 + threshold2) / 2

    f = open("results/price_distribution.txt", "a")
    f.write(type + ", window: " + str(window) + ", thirds margin: " + str(thirds_split) + "\n")
    f.close()

    print(thirds_split)

    return thirds_split


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
        margin = margin_range[0] + np.random.rand() * (margin_range[1] - margin_range[0])
        # margin = margin_range[0] + (i / n_iter) * (margin_range[1] - margin_range[0])
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
        plt.fill_between([major for _, _, major in scores], np.array([sc for sc, _, _ in scores]) - np.array([std for _, std, _ in scores]), np.array([sc for sc, _, _ in scores] + np.array([std for _, std, _ in scores])), alpha=0.5)
        # plt.errorbar([major for _, _, major in scores], [sc for sc, _, _ in scores], yerr=[std for _, std, _ in scores])
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

# TODO: what are best values for back_window sizes?
# build X matrix multiple times and find best sizes using regression

# ----------------------------------------------------------------------------------------------------


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
    threshold = kwargs["threshold"]
    ids = set(kwargs["IDs"])
    labels = kwargs["labels"]
    margin = kwargs["margin"]
    window = kwargs["window"]
    del kwargs

    dict_key = ""
    is_conversation = False
    if type == "articles":
        dict_key = "reduced_text"
    elif type == "conversations":
        dict_key = "clean_text"
        is_conversation = True
    elif type == "tweets":
        dict_key = "clean_text"

    tfidf_indexes = np.zeros(len(labels), dtype="bool")
    for i, label in enumerate(labels):
        if label.find(".") != -1:
            tfidf_indexes[i] = True

    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("fully weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    tfidf, vocabulary = common.calc_tf_idf(raw_data, min_df=0.0, max_df=1.0, dict_key=dict_key, is_conversation=is_conversation)
    remove_tfidf_lines = np.zeros(tfidf.shape[0], dtype="bool")
    if not is_conversation:
        for i, text in enumerate(raw_data):
            if text["_id"] not in ids:
                remove_tfidf_lines[i] = True
    else:
        i = 0
        for text in raw_data:
            for currency in text["coin_mentions"]:
                if not str(text["_id"]) + ":" + currency in ids:
                    remove_tfidf_lines[i] = True
                i += 1

    tfidf = tfidf[~remove_tfidf_lines, :]
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("non weighted - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    financial_weights = common.create_financial_vocabulary(vocabulary) + 1
    _f = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _f.setdiag(financial_weights)
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _f]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("only financial weights - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    sentiment_weights = common.calc_sentiment_for_vocabulary(vocabulary)
    sentiment_weights = np.where(sentiment_weights >= 0, sentiment_weights + 1, sentiment_weights - 1)
    _s = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _s.setdiag(sentiment_weights)
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _s]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("only sentiment weights - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    weights = sparse.csr_matrix(financial_weights * sentiment_weights)
    _w = sparse.lil_matrix((tfidf.shape[1], tfidf.shape[1]))
    _w.setdiag(weights.toarray()[0])
    data_X = data_X[:, ~tfidf_indexes]
    data_X = sparse.hstack([data_X, tfidf * _w]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/tfidf_test_scores.txt", "a")
    f.write("sentiment & financial weights - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.write("\n")
    f.close()

# TODO: if previous TF-IDFs contribute to accuracy, try different weights


def topics_test(**kwargs):
    print("TEST TOPICS")
    # topics - build X matrices:
    #   - without topics
    #   - with current topics only
    #   - with past topics only
    #   - with both sets of features

    type = kwargs["type"]
    train_f = kwargs["train_f"]
    data_X = kwargs["data_X"]
    data_Y = kwargs["data_Y"]
    n = kwargs["n"]
    model = kwargs["model"]
    feature_selector = kwargs["feature_selector"]
    threshold = kwargs["threshold"]
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

    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("full - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    average_topics = data_X[:, average_topic_indexes]
    data_X = data_X[:, ~average_topic_indexes]
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("topics only - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    data_X = data_X[:, ~topic_indexes[:data_X.shape[1]]]
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("none: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.close()

    data_X = sparse.hstack([data_X, average_topics]).tocsr()
    _, score, score_std, precision, recall, _, _ = train_f(n=n, feature_selector=feature_selector, model=model, data_X=data_X, data_Y=data_Y, type=type, threshold=threshold, save=False, train_seperate_set=False)
    f = open("results/topic_test_scores.txt", "a")
    f.write("average topics only - score: " + str(score) + " (+/- " + str(score_std) + "), precision: " + str(precision) + ", recall: " + str(recall) + ", margin: " + str(margin) + ", window: " + str(window) + "\n")
    f.write("\n")
    f.close()


# TODO: tabulate contribution coefficients for technical data.
# also, build X matrices:
#   - with past prices only
#   - with past volumes only
#   - with overall market movements only
#   - without any of them
#   - with all of them

# -----------------------------------------------------------------------------------------------------

# TODO: do grid search to find best params for every learning algorithm used


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

# TODO: naively calculate expected returns for time period (e.g. 1 year)

