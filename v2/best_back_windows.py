import numpy as np
import pymongo

import common as common
import time
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_selection import mutual_info_classif
from itertools import repeat
import matplotlib.pyplot as plt
import pickle

import news as news
import trollbox as trollbox
import twitter as twitter


def parallelized_matrix_creation(k, n_iter, back_window_range, raw_data, ids, type, date_key, currency_key, client):
    back_window = 300 * round((back_window_range[0] + ((k + 1) / n_iter) * (back_window_range[1] - back_window_range[0])) / 300)
    
    print("THREAD", k)
    matrix = None

    for i, text in enumerate(raw_data):
        print("iteration", i, "" if matrix is None else matrix.shape)
        if text["_id"] in ids:
            date_from = int(time.mktime(text[date_key].timetuple()) + 3600)
            matrix_line = []
            matrix_line += common.get_averages_from_data(raw_data, text[date_key], back_window, text[currency_key], i, threshold=0.0, type=type[:-1], data_averages_only=True)
            # matrix_line += common.get_averages_from_db(client, text[date_key], back_window, text[currency_key], articles=articles, conversations=conversations, tweets=tweets)

            matrix_line.append(common.get_price_change(client, text[currency_key], date_from - back_window, date_from))
            matrix_line.append(common.get_all_price_changes(client, date_from - back_window, date_from))

            volume_all = common.get_total_volume(client, "all", date_from - back_window, date_from)
            volume_curr = common.get_total_volume(client, text[currency_key], date_from - back_window, date_from)
            matrix_line.append(0 if volume_all == 0 else volume_curr / volume_all)

            matrix_line = np.where(np.isfinite(matrix_line), matrix_line, 0)

            if matrix is None:
                matrix = sparse.csr_matrix(matrix_line)
            else:
                matrix = sparse.vstack([matrix, sparse.csr_matrix(matrix_line)]).tocsr()

    labels = ["distribution", "polarity", "sentiment", "price", "price_all", "volume"]
    for i in range(len(labels)):
        labels[i] += "_" + str(back_window)

    return matrix, labels, back_window

    # return price_X, volume_X, price_all_X, distribution_1_X, distribution_2_X, distribution_3_X, sentiment_1_X, sentiment_2_X, sentiment_3_X, polarity_1_X, polarity_2_X, polarity_3_X, back_window


def get_mutual_info(X, Y, type, window, save):
    X = X.todense()
    mi = mutual_info_classif(X, Y, discrete_features=False, n_neighbors=3)
    mi = np.array(mi).reshape((-1, 6))
    mi = mi.mean(axis=1)

    if save:
        print(window)
        print(str(window))
        print(type)
        print(str(type))
        pickle.dump(mi, open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_mutual_info_for_" + str(window) + ".pickle", "wb"))
    
    return mi


def load_mutual_info(X, Y, window, type="articles"):
    try:
        f = open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_mutual_info_for_" + str(window) + ".pickle", "rb")
        pickle.load(f)
        f.close()
    except FileNotFoundError:
        X = X.todense()
        mi = mutual_info_classif(X, Y, discrete_features=False, n_neighbors=3)
        mi = np.array(mi).reshape((-1, 6))
        mi = mi.mean(axis=1)
        f = open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_mutual_info_for_" + str(window) + ".pickle", "wb")
        pickle.dump(mi, f)
        f.close()


def plot(mi, back_windows, window):
    to_plot = list(zip(back_windows, mi))

    back_windows = [back_window for back_window, _ in to_plot]
    mi = [_mi for _, _mi in to_plot]

    plt.plot(back_windows, mi)
    plt.savefig("figures/mutual_info_plot_" + str(window) + "_" + str(time.time()) + ".png")


def reverse_matrix(X):
    _X = None
    X = X.toarray().transpose()
    return sparse.csr_matrix(X)


def k_means(X, Y, window):
    f = open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + "articles" + "_back_windows.pickle", "rb")
    back_windows = pickle.load(f)
    f.close()
    back_windows = np.array(back_windows)
    
    mi = get_mutual_info(reverse_matrix(X), Y, "articles", window, False)
    mi_mask = np.where(mi >= np.mean(mi) - np.std(mi), True, False)
    print(np.mean(mi) - np.std(mi))
    
    X = X.toarray()
    select = np.array(list(range(0, 3000, 6)))
    
    select = select[mi_mask]
    back_windows = back_windows[mi_mask]
    
    for n_clusters in range(2, 3):
        silhouettes = []
        all_windows = []
        for i in range(6):
            # print(i)
            _X = X[select + i, :]
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=1, n_init=20)
            labels = kmeans.fit_predict(_X)
            clusters = Counter(labels).keys()
            windows = []
            for cluster in clusters:
                selection = back_windows[np.where(labels == cluster)]
                windows.append((np.mean(selection), np.std(selection)))
        
            windows.sort(key=lambda x: x[0])
            # for window in windows:
            #     print(window)
            silhouette = silhouette_score(_X, labels)
            silhouettes.append(silhouette)
            all_windows.append(windows)
            # print("silhouette score", silhouette)
        print(np.mean(all_windows, axis=0))
        print("NCLUSTERS", n_clusters, "SILHOUETTE SCORE", np.mean(silhouettes))
        
    opt_windows = np.mean(all_windows, axis=0)
    windows = []
    indexes = []
    for i in range(len(opt_windows)):
        window = int(300 * round(opt_windows[i][0] / 300))
        try:
            indexes.append(back_windows.tolist().index(window))
        except ValueError:
            window -= 300
            indexes.append(back_windows.tolist().index(window))
        windows.append(window)
            
    
    print(windows, indexes)
    return windows


def best_back_windows(**kwargs):
    print("FIND BEST BACK WINDOWS")
    n_iter = kwargs["n_iter"]
    n_iter = 500
    back_window_range = kwargs["back_window_range"]
    client = pymongo.MongoClient(host="127.0.0.1", port=27017)
    ids = set(kwargs["IDs"])
    raw_data = kwargs["raw_data"]
    type = kwargs["type"]
    final_set_f = kwargs["final_set_f"]
    window = kwargs["window"]
    margin = kwargs["margin"]
    del kwargs

    # ids = set(final_set_f(None, None, ids)[0])

    date_key = "date"
    currency_key = "currency"
    get_Y_f = news.get_Y

    pool = ThreadPool()
    results = pool.starmap(parallelized_matrix_creation, zip(list(range(n_iter)), repeat(n_iter), repeat(back_window_range), repeat(raw_data), repeat(ids), repeat(type), repeat(date_key), repeat(currency_key), repeat(client)))
    pool.close()
    pool.join()

    X = None
    labels = []
    back_windows = []
    for result in results:
        if X is None:
            X = sparse.csr_matrix(result[0])
        else:
            X = sparse.hstack([X, result[0]]).tocsr()
        labels += result[1]
        back_windows.append(result[2])
        print(X.shape, len(labels), len(back_windows))

    pickle.dump(X, open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_X_back_windows.pickle", "wb"))
    pickle.dump(labels, open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_labels_back_windows.pickle", "wb"))
    pickle.dump(back_windows, open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_back_windows.pickle", "wb"))

    Y = np.array(get_Y_f(ids, window, margin))
    X, Y, _, _, _ = final_set_f(X, Y)
    get_mutual_info(X, Y, type, window, True)
    plot(window)


def create_mutual_info(**kwargs):
    print("MUTUAL INFORMATION")
    type = "articles"
    final_set_f = kwargs["final_set_f"]
    get_Y_f = news.get_Y
    window = kwargs["window"]
    margin = kwargs["margin"]
    ids = set(kwargs["IDs"])
    
    back_windows = np.array(pickle.load(open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_back_windows.pickle", "rb")))
    X = pickle.load(open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_X_back_windows.pickle", "rb"))
    Y = np.array(get_Y_f(ids, window, margin))
    X, Y, _, _, _ = final_set_f(X, Y)

    # load_mutual_info(X, Y, window)
    mi = get_mutual_info(X, Y, "articles", window, False)
    mi = np.array(mi)
    better_than = np.mean(mi) + 2*np.std(mi)
    print(better_than)
    selected_windows = back_windows[np.where(mi >= better_than)]
    print(selected_windows)
    
    # plot(mi, back_windows, window)
    return selected_windows


def create_k_means(**kwargs):
    print("KMEANS")
    type = "articles"
    final_set_f = kwargs["final_set_f"]
    get_Y_f = news.get_Y
    window = kwargs["window"]
    margin = kwargs["margin"]
    ids = set(kwargs["IDs"])
    
    if "all_history_data" in kwargs:
        X = kwargs["all_history_data"]
    else:
        X = pickle.load(open("/home/ubuntu/diploma/Proletarian 1.0/v2/pickles/" + type + "_X_back_windows.pickle", "rb"))
    Y = np.array(get_Y_f(ids, window, margin))
    X, Y, _, _, _ = final_set_f(X, Y)

    X = reverse_matrix(X)
    return k_means(X, Y, window)

