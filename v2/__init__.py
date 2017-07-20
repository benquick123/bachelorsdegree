import v2.twitter as twitter
import v2.trollbox as trollbox
import v2.news as news
import v2.common as common
import v2.pickle_loading_saving as pls
from v2.simulator import simulate
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from collections import Counter
import numpy as np


def initial_load():
    news.load_without_attr(p=True)
    trollbox.load_without_attr(p=True)
    twitter.load_without_attr(p=True)


def train(n, feature_selector, model, data_X, data_Y, type, threshold=1, save=True, p=True):
    # ids = np.array(pickle_loading_saving.load_matrix_IDs(type))
    data_Y = np.array(data_Y)
    scores, precision, recall, earnings = [], [], [], []

    indexes = np.array(np.linspace(0, data_X.shape[0] - 1, n + 1), dtype="int")
    classes = dict(Counter(data_Y))
    if p:
        print("classes:", classes)

    for i in range(n):
        mask = np.array([True] * data_X.shape[0])
        mask[indexes[i]:indexes[i + 1]] = False

        sfm = SelectFromModel(feature_selector, threshold=str(threshold)+"*mean")
        sfm.fit(data_X[mask, :], data_Y[mask])
        reduced_data_X = sfm.transform(data_X)

        if p:
            print("features:", data_X.shape[1], "->", reduced_data_X.shape[1])
            print("training model")

        model.fit(reduced_data_X[mask, :], data_Y[mask])
        pred_Y = model.predict(reduced_data_X[~mask, :])

        scores.append(accuracy_score(data_Y[~mask], pred_Y))
        precision.append(precision_score(data_Y[~mask], pred_Y, average=None))
        recall.append(recall_score(data_Y[~mask], pred_Y, average=None))

        # true_Y += data_Y[~mask].tolist()
        # predicted_Y += pred_Y

    if p:
        print("accuracy: %0.3f (+/- %0.3f)" % (np.mean(scores), np.std(scores)))
        print("precision: %0.3f, recall: %0.3f" % (np.mean(precision), np.mean(recall)))
        # classification_report(data_Y[~mask], pred_Y)

    if save:
        f = open("results/" + type + "_results.txt", "a")
        s = ""
        s += "samples: " + str(reduced_data_X.shape[0]) + ", features: " + str(reduced_data_X.shape[1]) + "\n"
        s += "majority class: " + str(max(classes.values()) / reduced_data_X.shape[0])[:5] + "\n"
        s += "accuracy: " + str(np.mean(scores))[:5] + " (+/- " + str(np.std(scores))[:5] + ")\n"
        s += "precision: " + str(np.mean(precision))[:5] + ", recall: " + str(np.mean(recall))[:5] + "\n"
        s += "cv: " + str(n) + "\n"
        s += "feature selection: " + str(feature_selector).replace("\n", " ").replace("     ", "") + "\n"
        s += "model: " + str(model).replace("\n", " ").replace("     ", "") + "\n"
        f.write(s)
        f.close()

    return scores, precision, recall, model


def train_articles(window, margin, n=None, p=False, data=False, matrix=False, save=True):
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
        articles_X = pls.load_matrix_X("articles")
        articles_Y = pls.load_matrix_Y("articles", window, margin)
        if articles_Y is None:
            articles_Y = news.get_Y(IDs, window, margin)
            if articles_Y is not None and save:
                pls.save_matrix_Y(articles_Y, "articles", window, margin)
            elif articles_Y is None:
                exit()

    print("deleting articles")
    del articles

    features_threshold = 0.0
    while features_threshold < 3:
        feature_selector = LinearSVC()
        model = LinearSVC()
        n = 10
        train(n, feature_selector, model, articles_X, articles_Y, "articles", features_threshold, save)
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

    feature_selector = LinearSVC()
    model = RandomForestClassifier()
    n = 10
    train(n, feature_selector, model, conversations_X, conversations_Y, "conversations", save)
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

    print("deleting tweets")
    del tweets

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

    window = 1800
    margin = 0.004
    train_articles(window, margin, p=True, data=True, matrix=True, save=True)
    exit()

    # window = 1800
    # margin = 0.004
    # n_tweets = 100000
    # train_tweets(window, margin, n=n_tweets, p=True, data=True)
    # exit()

    window = 900
    margin = 0.01
    n_conversations = 200000
    train_conversations(window, margin, n=n_conversations, p=True, data=True)
    exit()

__init__()
