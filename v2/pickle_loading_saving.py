import pickle
from scipy import sparse


def save_data_pickle(data, type, k=None):
    print("saving data pickle")
    if k is None:
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_data.pickle", "wb")
        pickle.dump(data, f)
        f.close()
    else:
        i = 0
        while i + k <= len(data):
            f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_data_" + str(i) + ".pickle", "wb")
            pickle.dump(data[i:i + k], f)
            f.close()
            i += k
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_data_" + str(i) + ".pickle", "wb")
        pickle.dump(data[i:len(data)], f)
        f.close()


def load_data_pickle(type, k=None):
    print("loading data pickle")
    if k is None:
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_data.pickle", "rb")
        data = pickle.load(f)
        f.close()
    else:
        i = 0
        data = []
        try:
            while True:
                f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_data_" + str(i) + ".pickle", "rb")
                data += pickle.load(f)
                f.close()
                i += k
        except FileNotFoundError:
            pass
    return data


def save_matrix_X(data_X, type, k=None):
    print("saving matrix_X pickle")
    if k is None:
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_X.pickle", "wb")
        pickle.dump(data_X, f)
        f.close()
    else:
        i = 0
        while i + k <= data_X.shape[0]:
            f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_X_" + str(i) + ".pickle", "wb")
            pickle.dump(data_X[i:i + k, :], f)
            f.close()
            i += k
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_X_" + str(i) + ".pickle", "wb")
        pickle.dump(data_X[i:data_X.shape[0], :], f)
        f.close()


def load_matrix_X(type, k=None):
    print("loading matrix_X pickle")
    if k is None:
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_X.pickle", "rb")
        data_X = pickle.load(f)
        f.close()
    else:
        i = 0
        data_X = None
        try:
            while True:
                f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_X_" + str(i) + ".pickle", "rb")
                if data_X is None:
                    data_X = pickle.load(f)
                else:
                    data_X = sparse.vstack([data_X, pickle.load(f)])
                f.close()
                i += k
        except FileNotFoundError:
            pass
    return data_X


def save_matrix_Y(data_Y, type, window, margin):
    print("saving matrix_Y pickle")
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_Y_" + str(window) + "_" + str(margin) + ".pickle", "wb")
    pickle.dump(data_Y, f)
    f.close()


def load_matrix_Y(type, window, margin):
    print("loading matrix_Y pickle")
    try:
        f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_Y_" + str(window) + "_" + str(margin) + ".pickle", "rb")
        data_Y = pickle.load(f)
        f.close()
    except FileNotFoundError:
        data_Y = None
    return data_Y


def save_matrix_IDs(data_IDs, type):
    print("saving matrix_ID pickle")
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_ID.pickle", "wb")
    pickle.dump(data_IDs, f)
    f.close()


def load_matrix_IDs(type):
    print("loading matrix_ID pickle")
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_matrix_ID.pickle", "rb")
    data_IDs = pickle.load(f)
    f.close()
    return data_IDs


def save_labels(labels, type):
    print("saving matrix labels")
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_labels.pickle", "wb")
    pickle.dump(labels, f)
    f.close()


def load_labels(type):
    print("loading matrix labels")
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/pickles/" + type + "/" + type + "_labels.pickle", "rb")
    labels = pickle.load(f)
    f.close()
    return labels
