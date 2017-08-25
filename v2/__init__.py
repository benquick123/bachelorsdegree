from collections import Counter

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import scipy.sparse as sparse
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import copy


# import v2.common as common
import news as news
import pickle_loading_saving as pls
import trollbox as trollbox
import twitter as twitter
import test_code as test_code
import parameter_search as parameter_search
import best_back_windows as best_back_windows


def initial_load():
    news.load_without_attr(p=True)
    trollbox.load_without_attr(p=True)
    twitter.load_without_attr(p=True)


def create_final_test_set(data_X, data_Y, IDs=None):
    if IDs is None:
        r = np.zeros(data_X.shape[0], dtype="bool")
    else:
        r = np.zeros(len(IDs), dtype="bool")
    split = int(2 * r.shape[0] / 10)
    r[-split:] = True

    if IDs is not None:
        final_IDs = IDs[r]
        IDs = IDs[~r]
        return IDs, final_IDs, r

    final_test_X = data_X[r, :]
    final_test_Y = data_Y[r]
    data_X = data_X[~r, :]
    data_Y = data_Y[~r]

    return data_X, data_Y, final_test_X, final_test_Y, r


def get_dates_list(ids, data, type):
    dates = []
    date_key = ""
    if type == "articles":
        date_key = "date"
    elif type == "conversations":
        date_key = "conversation_end"
    elif type == "tweets":
        date_key = "posted_time"

    ids = set(ids)
    for text in data:
        if text["_id"] in ids:
            dates.append(text[date_key])

    return dates


def daily_split(data_X, dates, final_test):
    if final_test:
        testing_subset = np.zeros(data_X.shape[0], dtype="int")
    else:
        testing_subset = np.zeros(data_X.shape[0], dtype="int") - 1
        testing_indexes = int(2 * data_X.shape[0] / 10)
        testing_subset[-testing_indexes:] = 0

    old_day = 0
    for i in range(len(testing_subset)):
        new_day = dates[i].day
        if testing_subset[i] >= 0 and new_day != old_day:
            testing_subset[i:] += 1
        old_day = new_day

    # print(testing_subset.tolist())
    return testing_subset


def test_in_parallel(i, testing_indexes, data_X, data_Y, final_test_X, final_test_Y, feature_selector, model, p):
    train_mask = np.where(testing_indexes < i, True, False)
    test_mask = np.where(testing_indexes == i, True, False)

    if not np.any(test_mask):
        return None

    X = sparse.vstack([data_X, final_test_X[train_mask, :]])
    Y = np.concatenate((data_Y, final_test_Y[train_mask]))

    sfm = copy.deepcopy(feature_selector)
    sfm.fit(X, Y)
    reduced_data_X = sfm.transform(data_X)
    reduced_final_test_X = sfm.transform(final_test_X)

    if p:
        print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])

    model = copy.deepcopy(model)
    X = sparse.vstack([reduced_data_X, reduced_final_test_X[train_mask]])
    model.fit(X, Y)
    return model.predict(reduced_final_test_X[test_mask]).tolist(), reduced_final_test_X.shape


def train_in_parallel(i, testing_indexes, data_X, data_Y, feature_selector, model, p):
    train_mask = np.where(testing_indexes < i, True, False)
    test_mask = np.where(testing_indexes == i, True, False)

    sfm = copy.deepcopy(feature_selector)
    sfm.fit(data_X[train_mask, :], data_Y[train_mask])
    reduced_data_X = sfm.transform(data_X)

    if p:
        print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])
        print("training model", i)

    model = copy.deepcopy(model)
    model.fit(reduced_data_X[train_mask, :], data_Y[train_mask])
    pred_Y = model.predict(reduced_data_X[test_mask, :])

    return accuracy_score(data_Y[test_mask], pred_Y), precision_score(data_Y[test_mask], pred_Y, average="weighted"), recall_score(data_Y[test_mask], pred_Y, average="weighted"), reduced_data_X.shape


def train(feature_selector, model, data_X, data_Y, type, dates, save=True, p=True, learn=True, test=True):
    dates = np.array(dates)
    data_Y = np.array(data_Y)
    data_X, data_Y, final_test_X, final_test_Y, r = create_final_test_set(data_X, data_Y)
    final_test_dates = dates[r]
    dates = dates[~r]

    classes = dict(Counter(data_Y))

    if learn:
        scores, precisions, recalls, earnings = [], [], [], []
        testing_indexes = daily_split(data_X, dates, False)

        pool = ThreadPool()
        results = pool.starmap(train_in_parallel, zip(list(range(max(testing_indexes)+1)), itertools.repeat(testing_indexes), itertools.repeat(data_X), itertools.repeat(data_Y), itertools.repeat(feature_selector), itertools.repeat(model), itertools.repeat(p)))
        pool.close()
        pool.join()

        for result in results:
            scores.append(result[0])
            precisions.append(result[1])
            recalls.append(result[2])
        reduced_data_X_shape = results[-1][3]

        """for i in range(max(testing_indexes)+1):
            train_mask = np.where(testing_indexes < i, True, False)
            test_mask = np.where(testing_indexes == i, True, False)

            sfm = feature_selector
            sfm.fit(data_X[train_mask, :], data_Y[train_mask])
            reduced_data_X = sfm.transform(data_X)

            if p:
                print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])
                print("training model", i)

            model.fit(reduced_data_X[train_mask, :], data_Y[train_mask])
            pred_Y = model.predict(reduced_data_X[test_mask, :])

            scores.append(accuracy_score(data_Y[test_mask], pred_Y))
            # scores.append(model.score(reduced_data_X[~mask, :], data_Y[~mask]))
            precisions.append(precision_score(data_Y[test_mask], pred_Y, average="weighted"))
            recalls.append(recall_score(data_Y[test_mask], pred_Y, average="weighted"))"""

        if p:
            print("classes:", dict(Counter(data_Y)))
            print("accuracy: %0.3f (+/- %0.3f)" % (np.mean(scores), np.std(scores)))
            print("precision: %0.3f, recall: %0.3f" % (np.mean(precisions), np.mean(recalls)))

    if test:
        classes = dict(Counter(final_test_Y))
        if p:
            print("\ntraining seperate training set")
            print("classes:", classes)

        testing_indexes = daily_split(final_test_X, final_test_dates, True)
        pred_Y = []

        pool = ThreadPool()
        results = pool.starmap(test_in_parallel, zip(list(range(max(testing_indexes)+1)), itertools.repeat(testing_indexes), itertools.repeat(data_X), itertools.repeat(data_Y), itertools.repeat(final_test_X), itertools.repeat(final_test_Y), itertools.repeat(feature_selector), itertools.repeat(model), itertools.repeat(p)))
        pool.close()
        pool.join()

        for result in results:
            if result is not None:
                pred_Y += result
        reduced_final_test_X_shape = results[-1][1]

        """for i in range(max(testing_indexes)+1):
            train_mask = np.where(testing_indexes < i, True, False)
            test_mask = np.where(testing_indexes == i, True, False)

            if not np.any(test_mask):
                continue

            X = sparse.vstack([data_X, final_test_X[train_mask, :]])
            Y = np.concatenate((data_Y, final_test_Y[train_mask]))

            sfm = feature_selector
            sfm.fit(X, Y)
            reduced_data_X = sfm.transform(data_X)
            reduced_final_test_X = sfm.transform(final_test_X)
            if p:
                print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])

            X = sparse.vstack([reduced_data_X, reduced_final_test_X[train_mask]])
            model.fit(X, Y)
            pred_Y += model.predict(reduced_final_test_X[test_mask]).tolist()"""

        score = accuracy_score(final_test_Y, pred_Y)
        precision = precision_score(final_test_Y, pred_Y, average="weighted")
        recall = recall_score(final_test_Y, pred_Y, average="weighted")

        final_Y = label_binarize(final_test_Y, classes=[-1, 0, 1])
        pred_Y = label_binarize(pred_Y, classes=[-1, 0, 1])
        roc = roc_auc_score(final_Y, pred_Y, average="weighted")

        if p:
            print("accuracy: %0.3f" % score)
            print("precision: %0.3f, recall: %0.3f" % (precision, recall))

        if save:
            f = open("results/" + type + "_results.txt", "a")
            s = ""
            s += "samples: " + str(reduced_final_test_X_shape[0]) + ", features: " + str(reduced_final_test_X_shape[1]) + "\n"
            s += "days: " + str(len(Counter(testing_indexes))) + "\n"
            s += "majority class: " + str(max(classes.values()) / len(final_test_Y))[:5] + "\n"
            s += "accuracy: " + str(score)[:5] + "\n"
            s += "precision: " + str(precision)[:5] + ", recall: " + str(recall)[:5] + "\n"
            s += "area under roc: " + str(roc)
            if learn:
                s += "cv - accuracy: " + str(np.mean(scores))[:5] + " (+/- " + str(np.std(scores))[:5] + ")\n"
            s += "feature selection: " + str(feature_selector).replace("\n", "").replace("  ", "") + "\n"
            s += "model: " + str(model).replace("\n", "").replace("  ", "") + "\n"
            f.write(s)
            f.close()

    elif learn and save:
        classes = dict(Counter(data_Y))
        f = open("results/" + type + "_results.txt", "a")
        s = ""
        s += "samples: " + str(reduced_data_X_shape[0]) + ", features: " + str(reduced_data_X_shape[1]) + "\n"
        s += "majority class: " + str(max(classes.values()) / len(data_Y))[:5] + "\n"
        s += "accuracy: " + str(np.mean(scores))[:5] + " (+/- " + str(np.std(scores))[:5] + ")\n"
        s += "precision: " + str(np.mean(precisions))[:5] + ", recall: " + str(np.mean(recalls))[:5] + "\n"
        s += "days: " + str(len(Counter(testing_indexes))) + "\n"
        s += "feature selection: " + str(feature_selector).replace("\n", " ").replace("     ", "") + "\n"
        s += "model: " + str(model).replace("\n", " ").replace("     ", "") + "\n"
        f.write(s)
        f.close()

    if learn and not test:
        return model, np.mean(scores), np.std(scores), np.mean(precisions), np.mean(recalls), reduced_data_X_shape, classes
    elif test:
        return model, score, precision, recall, reduced_final_test_X_shape, classes

    return model


def train_articles(window, margin, n=None, p=False, data=False, matrix=False, save=True, functions=list()):
    print("NEWS")

    if not data:
        print("loading from database")
        articles = news.load_with_attr(n, p)
        if save:
            pls.save_data_pickle(articles, "articles")
    else:
        articles = pls.load_data_pickle("articles")
        if n is not None and n <= len(articles):
            articles = articles[:n]
        news.articles = articles

    if not matrix:
        print("creating matrix")
        articles_X, articles_Y, IDs, labels = news.create_matrix(articles, window, margin, p)
        if save:
            pls.save_matrix_X(articles_X, "articles")
            pls.save_matrix_Y(articles_Y, "articles", window, margin)
            pls.save_matrix_IDs(IDs, "articles")
            pls.save_labels(labels, "articles")
    else:
        IDs = pls.load_matrix_IDs("articles")
        labels = pls.load_labels("articles")
        articles_X = pls.load_matrix_X("articles")
        articles_Y = pls.load_matrix_Y("articles", window, margin)
        if articles_Y is None:
            articles_Y = news.get_Y(IDs, window, margin)
            if articles_Y is not None and save:
                pls.save_matrix_Y(articles_Y, "articles", window, margin)
            elif articles_Y is None:
                exit()

    threshold = 1.0
    feature_selector = SelectFromModel(LinearSVC(), threshold=str(threshold) + "*mean")
    # feature_selector = SelectPercentile(mutual_info_classif, 20)
    model = LinearSVC()

    if len(functions) != 0:
        arguments = dict()
        arguments["n_iter"] = 200
        arguments["threshold_range"] = [0.0, 3.0]
        arguments["margin_range"] = [0.0, 0.03]
        arguments["window_range"] = [300, 6*3600]
        arguments["back_window_short"] = [300, 3600]
        arguments["back_window_medium"] = [3900, 6*3600]
        arguments["back_window_long"] = [6*3600+300, 12*3600]
        arguments["back_window_range"] = [300, 12*3600]
        arguments["back_window_ratio"] = [0.5, 5]
        arguments["train_f"] = train
        arguments["final_set_f"] = create_final_test_set
        arguments["dates_f"] = get_dates_list
        arguments["feature_selector"] = feature_selector
        arguments["feature_selectors"] = [LinearSVC(), RandomForestClassifier()]
        arguments["model"] = model
        arguments["models"] = [LinearSVC(), RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier()]
        arguments["n"] = 10
        arguments["data_X"] = articles_X
        arguments["data_Y"] = articles_Y
        arguments["raw_data"] = articles
        arguments["type"] = "articles"
        arguments["IDs"] = IDs
        arguments["labels"] = labels
        arguments["window"] = window
        arguments["margin"] = margin
        arguments["threshold"] = threshold

        for f in functions:
            f(**arguments)

    else:
        dates = get_dates_list(set(IDs), articles, "articles")
        print("deleting articles")
        del articles
        del news.articles

        train(feature_selector, model, articles_X, articles_Y, "articles", dates, save=save, learn=True, test=False)

        if save:
            comment = input("comment: ")
            f = open("results/" + "articles" + "_results.txt", "a")
            s = ""
            s += "window: " + str(window) + ", threshold: " + str(margin) + "\n"
            s += "comment: " + comment.replace("\n", "") + "\n\n"
            f.write(s)
            f.close()


def train_conversations(window, margin, n=None, p=False, data=False, matrix=False, save=True, functions=list()):
    print("TROLLBOX")
    k = 20000

    if not data:
        print("loading from database")
        conversations = trollbox.load_with_attr(n, p)
        if save:
            pls.save_data_pickle(conversations, "conversations", k)
    else:
        conversations = pls.load_data_pickle("conversations", k)
        if n is not None and n <= len(conversations):
            conversations = conversations[:n]
        trollbox.conversations = conversations

    if not matrix:
        print("creating matrix")
        conversations_X, conversations_Y, IDs, labels = trollbox.create_matrix(conversations, window, margin, p)
        if save:
            pls.save_matrix_X(conversations_X, "conversations", k)
            pls.save_matrix_Y(conversations_Y, "conversations", window, margin)
            pls.save_matrix_IDs(IDs, "conversations")
            pls.save_labels(labels, "conversations")
    else:
        IDs = pls.load_matrix_IDs("conversations")
        labels = pls.load_labels("conversations")
        conversations_X = pls.load_matrix_X("conversations", k)
        conversations_Y = pls.load_matrix_Y("conversations", window, margin)
        if conversations_Y is None:
            conversations_Y = trollbox.get_Y(IDs, window, margin)
            if conversations_Y is not None and save:
                pls.save_matrix_Y(conversations_Y, "conversations", window, margin)
            else:
                exit()

    threshold = 1.0
    feature_selector = SelectFromModel(LinearSVC(), threshold=str(threshold)+"*mean")
    model = LinearSVC()

    if len(functions) != 0:
        arguments = dict()
        arguments["n_iter"] = 200
        arguments["threshold_range"] = [0.0, 3.0]
        arguments["margin_range"] = [0.0, 0.03]
        arguments["window_range"] = [300, 6 * 3600]
        arguments["back_window_short"] = [300, 3600]
        arguments["back_window_medium"] = [3900, 6 * 3600]
        arguments["back_window_long"] = [6 * 3600 + 300, 12 * 3600]
        arguments["back_window_range"] = [300, 12 * 3600]
        arguments["back_window_ratio"] = [0.5, 5]
        arguments["train_f"] = train
        arguments["final_set_f"] = create_final_test_set
        arguments["dates_f"] = get_dates_list
        arguments["feature_selector"] = feature_selector
        arguments["feature_selectors"] = [LinearSVC(), RandomForestClassifier()]
        arguments["model"] = model
        arguments["models"] = [LinearSVC(), RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier()]
        arguments["n"] = 10
        arguments["data_X"] = conversations_X
        arguments["data_Y"] = conversations_Y
        arguments["raw_data"] = conversations
        arguments["type"] = "conversations"
        arguments["IDs"] = IDs
        arguments["labels"] = labels
        arguments["window"] = window
        arguments["margin"] = margin
        arguments["threshold"] = threshold

        for f in functions:
            f(**arguments)

    else:
        dates = get_dates_list(IDs, conversations, "conversations")
        print("deleting conversations")
        del conversations
        del trollbox.conversations

        n = 10
        train(feature_selector, model, conversations_X, conversations_Y, "conversations", dates, save=save, learn=True, test=False)
        if save:
            comment = input("comment: ")
            f = open("results/" + "conversations" + "_results.txt", "a")
            s = ""
            s += "window: " + str(window) + ", threshold: " + str(margin) + "\n"
            s += "comment: " + comment.replace("\n", "") + "\n\n"
            f.write(s)
            f.close()


def train_tweets(window, margin, n=None, p=False, data=False, matrix=False, save=True, functions=list()):
    print("TWITTER")
    k = 20000

    if not data:
        print("loading from database")
        tweets = twitter.load_with_attr(n, p)
        if save:
            pls.save_data_pickle(tweets, "tweets", k)
    else:
        tweets = pls.load_data_pickle("tweets", k)
        if n is not None and n <= len(tweets):
            tweets = tweets[:n]
        twitter.tweets = tweets

    if not matrix:
        print("creating matrix")
        tweets_X, tweets_Y, IDs, labels = twitter.create_matrix(tweets, window, margin, p)
        if save:
            pls.save_matrix_X(tweets_X, "tweets", k)
            pls.save_matrix_Y(tweets_Y, "tweets", window, margin)
            pls.save_matrix_IDs(IDs, "tweets")
            pls.save_labels(labels, "tweets")
    else:
        IDs = pls.load_matrix_IDs("tweets")
        labels = pls.load_labels("tweets")
        tweets_X = pls.load_matrix_X("tweets", k)
        tweets_Y = pls.load_matrix_Y("tweets", window, margin)
        if tweets_Y is None:
            tweets_Y = twitter.get_Y(IDs, window, margin)
            if tweets_Y is not None and save:
                pls.save_matrix_Y(tweets_Y, "tweets", window, margin)
            elif tweets_Y is None:
                exit()

    threshold = 1.0
    feature_selector = SelectFromModel(LinearSVC(), threshold=str(threshold)+"*mean")
    model = LinearSVC()
    if len(functions) != 0:
        arguments = dict()
        arguments["n_iter"] = 200
        arguments["threshold_range"] = [0.0, 3.0]
        arguments["margin_range"] = [0.0, 0.03]
        arguments["window_range"] = [300, 6 * 3600]
        arguments["back_window_short"] = [300, 3600]
        arguments["back_window_medium"] = [3900, 6 * 3600]
        arguments["back_window_long"] = [6 * 3600 + 300, 12 * 3600]
        arguments["back_window_range"] = [300, 12 * 3600]
        arguments["back_window_ratio"] = [0.5, 5]
        arguments["train_f"] = train
        arguments["final_set_f"] = create_final_test_set
        arguments["dates_f"] = get_dates_list
        arguments["feature_selector"] = feature_selector
        arguments["feature_selectors"] = [LinearSVC(), RandomForestClassifier()]
        arguments["model"] = model
        arguments["models"] = [LinearSVC(), RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier()]
        arguments["n"] = 10
        arguments["data_X"] = tweets_X
        arguments["data_Y"] = tweets_Y
        arguments["raw_data"] = tweets
        arguments["type"] = "tweets"
        arguments["IDs"] = IDs
        arguments["labels"] = labels
        arguments["window"] = window
        arguments["margin"] = margin
        arguments["threshold"] = threshold

        for f in functions:
            f(**arguments)

    else:
        dates = get_dates_list(IDs, tweets, "tweets")
        print("deleting tweets")
        del tweets
        del twitter.tweets

        n = 10
        train(feature_selector, model, tweets_X, tweets_Y, "tweets", dates, save=save, learn=True, test=False)
        if save:
            comment = input("comment: ")
            f = open("results/" + "tweets" + "_results.txt", "a")
            s = ""
            s += "window: " + str(window) + ", threshold: " + str(margin) + "\n"
            s += "comment: " + comment.replace("\n", "") + "\n\n"
            f.write(s)
            f.close()


def __init__():
    # initial_load()

    functions = [parameter_search.randomized_data_params_search]

    window = 6600
    margin = 0.00968
    train_articles(window, margin, p=True, data=True, matrix=True, functions=functions)
    exit()

    window = 3*3600
    margin = 0.017
    train_tweets(window, margin, p=True, data=True, matrix=True, functions=[])
    # exit()

    window = 900
    margin = 0.005
    n_conversations = 150000
    train_conversations(window, margin, n=n_conversations, p=True, data=True, matrix=True, functions=functions)
    # exit()

if __name__ == "__main__":
    __init__()
