import v2.twitter as twitter
import v2.trollbox as trollbox
import v2.news as news
import v2.common as common
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from collections import Counter
from scipy import sparse
import pickle


def initial_load():
    news.load_without_attr(p=True)
    trollbox.load_without_attr(p=True)
    twitter.load_without_attr(p=True)


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

    print("classes:", dict(Counter(articles_Y)))
    print("training model")
    articles_model = LinearSVC()
    article_scores = cross_val_score(articles_model, articles_X, articles_Y, cv=20)
    print("accuracy: %0.3f (+/- %0.3f)" % (article_scores.mean(), article_scores.std()))


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
        del conversations
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

    print("classes:", dict(Counter(conversations_Y)))
    print("training model")
    conversations_model = LinearSVC()
    conversation_scores = cross_val_score(conversations_model, conversations_X, conversations_Y, cv=50)
    print("accuracy: %0.3f (+/- %0.3f)" % (conversation_scores.mean(), conversation_scores.std()))


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

    print("classes:", dict(Counter(tweets_Y)))
    print("training model")
    tweets_model = LinearSVC()
    tweet_scores = cross_val_score(tweets_model, tweets_X, tweets_Y, cv=50)
    print("accuracy: %0.3f (+/- %0.3f)" % (tweet_scores.mean(), tweet_scores.std()))


def __init__():
    # initial_load()

    # window = 3600 * 24
    # margin = 0.0
    # train_articles(window, margin, p=True, data=True)

    # window = 3600
    # margin = 0.0
    # n_tweets = 100000
    # train_tweets(window, margin, n=n_tweets, p=True, data=True)
    # exit()

    window = 3600
    margin = 0.025
    n_conversations = 50000
    train_conversations(window, margin, n=n_conversations, p=True, data=True)
    exit()

__init__()
