import v2.pickle_loading_saving as pls
<<<<<<< HEAD


def get_important_attrs():
    
=======
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, SelectPercentile, SelectFromModel, mutual_info_classif


def create_final_test_set(data_X, data_Y, IDs=None):
    np.random.seed(0)
    if IDs is None:
        r = np.array(np.random.choice(2, data_X.shape[0], p=[0.9, 0.1]), dtype="bool")
    else:
        r = np.array(np.random.choice(2, len(IDs), p=[0.9, 0.1]), dtype="bool")
    np.random.seed(None)

    if IDs is not None:
        final_IDs = IDs[r]
        IDs = IDs[~r]
        return IDs, final_IDs

    final_test_X = data_X[r, :]
    final_test_Y = data_Y[r]
    data_X = data_X[~r, :]
    data_Y = data_Y[~r]

    return data_X, data_Y, final_test_X, final_test_Y


def get_important_attrs():
    window = 1800
    margin = 0.004
    labels = pls.load_labels("tweets")
    X = pls.load_matrix_X("tweets", k=20000)
    Y = np.array(pls.load_matrix_Y("tweets", window, margin))
    X, Y, _, _ = create_final_test_set(X, Y)

    model = RandomForestClassifier()
    feature_selector = SelectPercentile(mutual_info_classif, 20)
    feature_selector = SelectFromModel(LinearSVC())
    model.fit(X, Y)

    # coef = np.mean(model.coef_, axis=1)
    # print(coef.shape)
    weights = []
    for label, coef in zip(labels, model.feature_importances_):
        weights.append((label, abs(coef)))
    # weights = zip(labels, model.coef_[0])
    # print(weights)
    weights.sort(key=lambda x: x[1])

    for w in weights:
        print(w)

get_important_attrs()
>>>>>>> d10c4a142d706e11a8525e5303be17ae11a4220d
