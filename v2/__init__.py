import v2.twitter as twitter
import v2.trollbox as trollbox
import v2.news as news
import v2.common as common
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier


def initial_load():
    conversations = trollbox.load_without_attr(p=True)
    tweets = twitter.load_without_attr(p=True)
    articles = news.load_without_attr(p=True)
    return conversations, tweets, articles


def __init__():
    window = 24 * 3600
    margin = 0.08

    # conversations, tweets, articles = initial_load()

    print("loading from database")
    n_conversations = 400000
    conversations = trollbox.load_with_attr(n_conversations, p=True)
    print("creating matrix")
    conversations_X, conversations_Y = trollbox.create_matrix(conversations, window, margin, p=True)
    print("deleting conversations")
    del conversations
    print("training model")
    conversations_model = LinearSVC()
    conversation_scores = cross_val_score(conversations_model, conversations_X, conversations_Y, cv=50)
    print("Accuracy: %0.3f (+/- %0.3f)" % (conversation_scores.mean(), conversation_scores.std()))

    print("loading from database")
    n_tweets = 300000
    margin = 0.04
    tweets = twitter.load_with_attr(n_tweets)
    print("creating matrix")
    tweets_X, tweets_Y = twitter.create_matrix(tweets, window, margin)
    print("deleting tweets")
    del tweets
    print("training model")
    tweets_model = LinearSVC()
    tweet_scores = cross_val_score(tweets_model, tweets_X, tweets_Y, cv=50)
    print("Accuracy: %0.3f (+/- %0.3f)" % (tweet_scores.mean(), tweet_scores.std()))

    """print("loading from database")
    articles = news.load_with_attr()
    print("creating matrix")
    articles_X, articles_Y = news.create_matrix(articles, window, margin)
    print("deleting articles")
    del articles
    print("training model")
    articles_model = LinearSVC()
    article_scores = cross_val_score(articles_model, articles_X, articles_Y, cv=50)
    print("Accuracy: %0.3f (+/- %0.3f)" % (article_scores.mean(), article_scores.std()))"""
    exit()

__init__()
