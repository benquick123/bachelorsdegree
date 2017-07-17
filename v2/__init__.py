import v2.twitter as twitter
import v2.trollbox as trollbox
import v2.news as news
import v2.common as common
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy import sparse
import pickle
import numpy as np


def initial_load():
    news.load_without_attr(p=True)
    trollbox.load_without_attr(p=True)
    twitter.load_without_attr(p=True)


def train(n, feature_selector, model, data_X, data_Y, p=True):
    data_Y = np.array(data_Y)
    scores, precision, recall = [], [], []
    indexes = np.array(np.linspace(0, data_X.shape[0] - 1, n + 1), dtype="int")
    for i in range(n):
        mask = np.array([True] * data_X.shape[0])
        mask[indexes[i]:indexes[i + 1]] = False
        if p:
            print(data_X.shape)
        sfm = SelectFromModel(feature_selector)
        sfm.fit(data_X[mask, :], data_Y[mask])
        reduced_data_X = sfm.transform(data_X)
        if p:
            print(reduced_data_X.shape)
            print("classes:", dict(Counter(data_Y)))
            print("training model")

        # conversation_scores = cross_val_score(conversations_model, conversations_X, conversations_Y, cv=10)
        model.fit(reduced_data_X[mask, :], data_Y[mask])
        pred_Y = model.predict(reduced_data_X[~mask, :])
        scores.append(accuracy_score(data_Y[~mask], pred_Y))
        precision.append(precision_score(data_Y[~mask], pred_Y, average="weighted"))
        recall.append(recall_score(data_Y[~mask], pred_Y, average="weighted"))

    # print("accuracy: %0.3f (+/- %0.3f)" % (conversation_scores.mean(), conversation_scores.std()))
    print("accuracy: %0.3f (+/- %0.3f)" % (np.mean(scores), np.std(scores)))
    print("precision: %0.3f, recall: %0.3f" % (np.mean(precision), np.mean(recall)))
    return scores, precision, recall, model


def train_articles(window, margin, n=None, p=False, data=False, matrix=False, save=True):
    print("NEWS")

    if not data:
        print("loading from database")
        articles = news.load_with_attr(n, p)
        if save:
            print("saving data pickle")
            f = open("pickles/articles/articles_data.pickle", "wb")
            pickle.dump(articles, f)
            f.close()
    else:
        print("loading data pickle")
        f = open("pickles/articles/articles_data.pickle", "rb")
        articles = pickle.load(f)
        f.close()

        if n is not None and n <= len(articles):
            articles = articles[:n]
        news.articles = articles

    if not matrix:
        print("creating matrix")
        articles_X, articles_Y = news.create_matrix(articles, window, margin, p)
        print("deleting articles")
        del articles
        if save:
            print("saving matrix pickle")
            f = open("pickles/articles/articles_matrix_X.pickle", "wb")
            pickle.dump(articles_X, f)
            f.close()
            f = open("pickles/articles/articles_matrix_Y.pickle", "wb")
            pickle.dump(articles_Y, f)
            f.close()
    else:
        print("loading matrix pickle")
        f = open("pickles/articles/articles_matrix_X.pickle", "rb")
        articles_X = pickle.load(f)
        f.close()
        f = open("pickles/articles/articles_matrix_Y.pickle", "rb")
        articles_Y = pickle.load(f)
        f.close()

    # _articles_X = sparse.csr_matrix(StandardScaler(copy=False).fit_transform(articles_X[:, :16].todense()))
    # articles_X[:, :16] = _articles_X
    # articles_X = StandardScaler(copy=False, with_mean=False).fit_transform(articles_X)

    feature_selector = LinearSVC()
    model = LinearSVC()
    n = 10
    train(n, feature_selector, model, articles_X, articles_Y)


def train_conversations(window, margin, n=None, p=False, data=False, matrix=False, save=True):
    print("TROLLBOX")
    k = 20000

    if not data:
        print("loading from database")
        conversations = trollbox.load_with_attr(n, p)
        if save:
            print("saving data pickle")
            i = 0
            while i+k <= len(conversations):
                f = open("pickles/conversations/conversations_data_" + str(i) + ".pickle", "wb")
                pickle.dump(conversations[i:i+k], f)
                f.close()
                i += k
            f = open("pickles/conversations/conversations_data_" + str(i) + ".pickle", "wb")
            pickle.dump(conversations[i:len(conversations)], f)
            f.close()
    else:
        print("loading data pickle")
        i = 0
        conversations = []
        try:
            while True:
                f = open("pickles/conversations/conversations_data_" + str(i) + ".pickle", "rb")
                conversations += pickle.load(f)
                f.close()
                i += k
        except FileNotFoundError:
            pass

        if n is not None and n <= len(conversations):
            conversations = conversations[:n]
        trollbox.conversations = conversations

    if not matrix:
        print("creating matrix")
        conversations_X, conversations_Y = trollbox.create_matrix(conversations, window, margin, p)
        print("deleting conversations")
        if save:
            print("saving matrix pickle")
            i = 0
            while i+k <= conversations_X.shape[0]:
                f = open("pickles/conversations/conversations_matrix_X_" + str(i) + ".pickle", "wb")
                pickle.dump(conversations_X[i:i+k, :], f)
                f.close()
                i += k
            f = open("pickles/conversations/conversations_matrix_X_" + str(i) + ".pickle", "wb")
            pickle.dump(conversations_X[i:conversations_X.shape[0], :], f)
            f.close()

            f = open("pickles/conversations/conversations_matrix_Y.pickle", "wb")
            pickle.dump(conversations_Y, f)
            f.close()
    else:
        print("loading matrix pickle")
        i = 0
        conversations_X = None
        try:
            while True:
                f = open("pickles/conversations/conversations_matrix_X_" + str(i) + ".pickle", "rb")
                if conversations_X is None:
                    conversations_X = pickle.load(f)
                else:
                    conversations_X = sparse.vstack([conversations_X, pickle.load(f)])
                f.close()
                i += k
        except FileNotFoundError:
            pass

        f = open("pickles/conversations/conversations_matrix_Y.pickle", "rb")
        conversations_Y = pickle.load(f)
        f.close()

    del conversations

    # _conversations_X = sparse.csr_matrix(StandardScaler(copy=False).fit_transform(conversations_X[:, :15].todense()))
    # conversations_X[:, :15] = _conversations_X
    # conversations_X = StandardScaler(copy=False, with_mean=False).fit_transform(conversations_X)

    feature_selector = LinearSVC()
    model = LinearSVC()
    n = 10
    train(n, feature_selector, model, conversations_X, conversations_Y)


def train_tweets(window, margin, n=None, p=False, data=False, matrix=False, save=True):
    print("TWITTER")
    k = 20000

    if not data:
        print("loading from database")
        tweets = twitter.load_with_attr(n, p)
        if save:
            print("saving data pickle")
            i = 0
            while i+k <= len(tweets):
                f = open("pickles/tweets/tweets_data_" + str(i) + ".pickle", "wb")
                pickle.dump(tweets[i:i+k], f)
                f.close()
                i += k
            f = open("pickles/tweets/tweets_data_" + str(i) + ".pickle", "wb")
            pickle.dump(tweets[i:len(tweets)], f)
            f.close()
    else:
        print("loading data pickle")
        i = 0
        tweets = []
        try:
            while True:
                f = open("pickles/tweets/tweets_data_" + str(i) + ".pickle", "rb")
                tweets += pickle.load(f)
                f.close()
                i += k
        except FileNotFoundError:
            pass

        if n is not None and n <= len(tweets):
            tweets = tweets[:n]
        twitter.tweets = tweets

    if not matrix:
        print("creating matrix")
        tweets_X, tweets_Y = twitter.create_matrix(tweets, window, margin, p)
        print("deleting tweets")
        del tweets
        if save:
            print("saving matrix pickle")
            i = 0
            while i+k <= tweets_X.shape[0]:
                f = open("pickles/tweets/tweets_matrix_X_" + str(i) + ".pickle", "wb")
                pickle.dump(tweets_X[i:i+k, :], f)
                f.close()
                i += k
            f = open("pickles/tweets/tweets_matrix_X_" + str(i) + ".pickle", "wb")
            pickle.dump(tweets_X[i:tweets_X.shape[0], :], f)
            f.close()

            f = open("pickles/tweets/tweets_matrix_Y.pickle", "wb")
            pickle.dump(tweets_Y, f)
            f.close()
    else:
        print("loading matrix pickle")
        i = 0
        tweets_X = None
        try:
            while True:
                f = open("pickles/tweets/tweets_matrix_X_" + str(i) + ".pickle", "rb")
                if tweets_X is None:
                    tweets_X = pickle.load(f)
                else:
                    tweets_X = sparse.vstack([tweets_X, pickle.load(f)])
                f.close()
                i += k
        except FileNotFoundError:
            pass

        f = open("pickles/tweets/tweets_matrix_Y.pickle", "rb")
        tweets_Y = pickle.load(f)
        f.close()

    feature_selector = LinearSVC()
    model = LinearSVC()
    n = 10
    train(n, feature_selector, model, tweets_X, tweets_Y)


def __init__():
    # initial_load()

    # window = 1800
    # margin = 0.005
    # train_articles(window, margin, p=True, data=True)
    # exit()

    # window = 900
    # margin = 0.004
    # n_tweets = 100000
    # train_tweets(window, margin, n=n_tweets, p=True, data=True)
    # exit()

    window = 900
    margin = 0.01
    n_conversations = 200000
    train_conversations(window, margin, n=n_conversations, p=True, data=True, matrix=True)
    exit()

__init__()
