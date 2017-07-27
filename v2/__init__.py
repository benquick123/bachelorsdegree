from collections import Counter

import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import mord

# import v2.common as common
import v2.news as news
import v2.pickle_loading_saving as pls
import v2.trollbox as trollbox
import v2.twitter as twitter
import v2.test_code as test_code


def initial_load():
    news.load_without_attr(p=True)
    trollbox.load_without_attr(p=True)
    twitter.load_without_attr(p=True)


def create_final_test_set(data_X, data_Y):
    np.random.seed(0)
    r = np.array(np.random.choice(2, data_X.shape[0], p=[0.9, 0.1]), dtype="bool")
    np.random.seed(None)
    final_test_X = data_X[r, :]
    final_test_Y = data_Y[r]
    data_X = data_X[~r, :]
    data_Y = data_Y[~r]
    return data_X, data_Y, final_test_X, final_test_Y


def train(n, feature_selector, model, data_X, data_Y, type, threshold=1.0, save=True, p=True, cross_validate=True, train_seperate_set=True):
    data_Y = np.array(data_Y)
    data_X, data_Y, final_test_X, final_test_Y = create_final_test_set(data_X, data_Y)

    if cross_validate:
        scores, precisions, recalls, earnings = [], [], [], []
        indexes = np.array(np.linspace(0, data_X.shape[0] - 1, n + 1), dtype="int")

        for i in range(n):
            mask = np.array([True] * data_X.shape[0])
            mask[indexes[i]:indexes[i + 1]] = False

            sfm = SelectFromModel(feature_selector)
            sfm.fit(data_X[mask, :], data_Y[mask])
            reduced_data_X = sfm.transform(data_X)

            if p:
                print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])
                print("training model")

            model.fit(reduced_data_X[mask, :], data_Y[mask])
            pred_Y = model.predict(reduced_data_X[~mask, :])

            scores.append(accuracy_score(data_Y[~mask], pred_Y))
            precisions.append(precision_score(data_Y[~mask], pred_Y, average="weighted"))
            recalls.append(recall_score(data_Y[~mask], pred_Y, average="weighted"))

        if p:
            print("classes:", dict(Counter(data_Y)))
            print("accuracy: %0.3f (+/- %0.3f)" % (np.mean(scores), np.std(scores)))
            print("precision: %0.3f, recall: %0.3f" % (np.mean(precisions), np.mean(recalls)))

    classes = dict(Counter(data_Y))
    if train_seperate_set:
        classes = dict(Counter(final_test_Y))
        if p:
            print("\ntraining seperate training set")
            print("classes:", classes)
        sfm = SelectFromModel(feature_selector)
        sfm.fit(data_X, data_Y)
        reduced_data_X = sfm.transform(data_X)
        reduced_final_test_X = sfm.transform(final_test_X)
        if p:
            print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])

        model.fit(reduced_data_X, data_Y)
        pred_Y = model.predict(reduced_final_test_X)

        score = accuracy_score(final_test_Y, pred_Y)
        precision = precision_score(final_test_Y, pred_Y, average="weighted")
        recall = recall_score(final_test_Y, pred_Y, average="weighted")

        final_test_Y = label_binarize(final_test_Y, classes=[-1, 0, 1])
        pred_Y = label_binarize(pred_Y, classes=[-1, 0, 1])
        roc = roc_auc_score(final_test_Y, pred_Y, average=None)

        if p:
            print("accuracy: %0.3f" % score)
            print("precision: %0.3f, recall: %0.3f" % (np.mean(precision), np.mean(recall)))

        if save:
            f = open("results/" + type + "_results.txt", "a")
            s = ""
            s += "samples: " + str(reduced_final_test_X.shape[0]) + ", features: " + str(reduced_final_test_X.shape[1]) + "\n"
            s += "majority class: " + str(max(classes.values()) / len(final_test_Y))[:5] + "\n"
            s += "accuracy: " + str(score)[:5] + "\n"
            s += "precision: " + str(precision)[:5] + ", recall: " + str(recall)[:5] + "\n"
            s += "area under roc: " + str(roc)
            if cross_validate:
                s += "cv: " + str(n) + ", accuracy: " + str(np.mean(scores))[:5] + " (+/- " + str(np.std(scores))[:5] + ")\n"
            s += "feature selection: " + str(feature_selector).replace("\n", "").replace("  ", "") + "\n"
            s += "model: " + str(model).replace("\n", "").replace("  ", "") + "\n"
            f.write(s)
            f.close()

    elif cross_validate and save:
        classes = dict(Counter(data_Y))
        f = open("results/" + type + "_results.txt", "a")
        s = ""
        s += "samples: " + str(reduced_data_X.shape[0]) + ", features: " + str(reduced_data_X.shape[1]) + "\n"
        s += "majority class: " + str(max(classes.values()) / len(data_Y))[:5] + "\n"
        s += "accuracy: " + str(np.mean(scores))[:5] + " (+/- " + str(np.std(scores))[:5] + ")\n"
        s += "precision: " + str(np.mean(precisions))[:5] + ", recall: " + str(np.mean(recalls))[:5] + "\n"
        s += "cv: " + str(n) + "\n"
        s += "feature selection: " + str(feature_selector).replace("\n", " ").replace("     ", "") + "\n"
        s += "model: " + str(model).replace("\n", " ").replace("     ", "") + "\n"
        f.write(s)
        f.close()

    if cross_validate and not train_seperate_set:
        return model, np.mean(scores), np.std(scores), np.mean(precisions), np.mean(recalls), reduced_data_X.shape, classes
    elif train_seperate_set:
        return model, score, precision, recall, roc, reduced_final_test_X.shape, classes

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

    feature_selector = LinearSVC()
    model = LinearSVC()
    if len(functions) != 0:
        arguments = dict()
        arguments["n_iter"] = 50
        arguments["threshold_range"] = [0.0, 3.0]
        arguments["margin_range"] = [0.0, 0.04]
        arguments["window_range"] = [300, 6*3600]
        arguments["train_f"] = train
        arguments["feature_selector"] = feature_selector
        arguments["feature_selectors"] = [LinearSVC(), RandomForestClassifier(), KNeighborsClassifier(), MLPClassifier()]
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
        arguments["threshold"] = 1.0

        for f in functions:
            f(**arguments)

    else:
        print("deleting articles")
        del articles
        del news.articles

        features_threshold = 1.4

        n = 10
        train(n, feature_selector, model, articles_X, articles_Y, "articles", threshold=features_threshold, save=save)
        features_threshold += 0.2

        if save:
            comment = input("comment: ")
            f = open("results/" + "articles" + "_results.txt", "a")
            s = ""
            s += "window: " + str(window) + ", threshold: " + str(margin) + "\n"
            s += "comment: " + comment.replace("\n", "") + "\n\n"
            f.write(s)
            f.close()


def train_conversations(window, margin, n=None, p=False, data=False, matrix=False, save=True):
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
        conversations_X = pls.load_matrix_X("conversations", k)
        conversations_Y = pls.load_matrix_Y("conversations", window, margin)
        if conversations_Y is None:
            conversations_Y = trollbox.get_Y(IDs, window, margin)
            if conversations_Y is not None and save:
                pls.save_matrix_Y(conversations_Y, "conversations", window, margin)
            else:
                exit()

    print("deleting conversations")
    del conversations
    del trollbox.conversations

    feature_selector = LinearSVC()
    model = LinearSVC()
    n = 10
    train(n, feature_selector, model, conversations_X, conversations_Y, "conversations", save=save)
    if save:
        comment = input("comment: ")
        f = open("results/" + "conversations" + "_results.txt", "a")
        s = ""
        s += "window: " + str(window) + ", threshold: " + str(margin) + "\n"
        s += "comment: " + comment.replace("\n", "") + "\n\n"
        f.write(s)
        f.close()


def train_tweets(window, margin, n=None, p=False, data=False, matrix=False, save=True):
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
        tweets_X = pls.load_matrix_X("tweets", k)
        tweets_Y = pls.load_matrix_Y("tweets", window, margin)
        if tweets_Y is None:
            tweets_Y = tweets.get_Y(IDs, window, margin)
            if tweets_Y is not None and save:
                pls.save_matrix_Y(tweets_Y, "tweets", window, margin)
            elif tweets_Y is None:
                exit()

    print("deleting tweets")
    del tweets
    del twitter.tweets

    feature_selector = LinearSVC()
    model = LinearSVC()
    n = 10
    train(n, feature_selector, model, tweets_X, tweets_Y, "tweets", save)
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

    functions = [test_code.tfidf_test]

    # proposed best for window=1800: margin = 0.00351643235977
    # best margin by optimal_margin = 0.004
    window = 1800
    margin = 0.004
    train_articles(window, margin, p=True, data=True, matrix=True, functions=functions)
    exit()

    window = 1800
    margin = 0.004
    n_tweets = 100000
    train_tweets(window, margin, n=n_tweets, p=True, data=True, matrix=True)
    exit()

    window = 900
    margin = 0.005
    n_conversations = 150000
    train_conversations(window, margin, n=n_conversations, p=True, data=True, matrix=True, save=False)
    exit()

__init__()
