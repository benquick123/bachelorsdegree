import matplotlib.pyplot as plt
import numpy as np
import re
from v2 import news
from v2 import trollbox
from v2 import twitter
from v2 import pickle_loading_saving
import scipy.stats as stats
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate


def plot_majority_class_corr():
    results_number = 1
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/results/majority_class_scores.txt", "r")
    s = f.read()
    f.close()
    occurrences = [m.start() for m in re.finditer("\n\n", s)]
    s = s[occurrences[results_number]:].replace("\n\n", "").split("\n")[1:]

    margins = []
    scores = []
    score_stds = []
    precisions = []
    recalls = []
    classes = []
    for _s in s:
        _s = _s.split(" ")
        margins.append(float(_s[3].replace(",", "")))
        scores.append(float(_s[5]))
        score_stds.append(float(_s[7].replace("),", "")))
        precisions.append(float(_s[9].replace(",", "")))
        recalls.append(float(_s[11].replace(",", "")))
        _classes = {}
        for i in range(13, len(_s), 2):
            _classes[int(_s[i].replace(":", "").replace("{", ""))] = int(_s[i+1].replace(",", "").replace("}", ""))
        classes.append(_classes)

    to_plot = []
    for i in range(len(scores)):
        major = max(classes[i].values()) / sum(classes[i].values())
        if margins[i] > 0.017:
            to_plot.append((scores[i] - major, score_stds[i], major))

    to_plot.sort(key=lambda x: x[2])
    plt.plot([major for _, _, major in to_plot], [sc for sc, _, _ in to_plot], "-", label="klasifikacijska točnost")
    plt.fill_between([major for _, _, major in to_plot], np.array([sc for sc, _, _ in to_plot]) - np.array([std for _, std, _ in to_plot]), np.array([sc for sc, _, _ in to_plot] + np.array([std for _, std, _ in to_plot])), alpha=0.5, label="standardni odklon")
    plt.axhline(y=0)
    plt.ylabel("klasifikacijska točnost")
    plt.xlabel("večinski razred")
    plt.legend(loc=3)
    plt.show()


def plot_window_corr():
    results_number = 1
    f = open("C:/Users/benqu/Documents/MEGA/Diplomsko-delo/Proletarian 1.0/v2/results/predict_back_window_scores_articles.txt", "r")
    s = f.read()
    f.close()
    occurrences = [m.start() for m in re.finditer("\n\n", s)]
    # s = s[:occurrences[results_number]].replace("\n\n", "").split("\n")[1:]
    # s = s[occurrences[results_number]:occurrences[results_number+1]].replace("\n\n", "").split("\n")[1:]    # + s[occurrences[results_number+1]:occurrences[results_number+2]].replace("\n\n", "").split("\n")[1:]
    s = s.replace("\n\n", "").split("\n")[1:-1]
    print(len(s))

    windows = []
    back_windows = []
    scores = []
    for _s in s:
        _s = _s.split(" ")
        scores.append(float(_s[3].replace(",", "")))
        back_windows.append(int(_s[5].replace(",", "")))
        windows.append(int(_s[7].replace(",", "")))

    scores = np.array(scores)
    windows = np.array(windows)
    back_windows = np.array(back_windows)
    ratios = back_windows / windows

    x_plot = np.linspace(0, max(scores), 500)
    z = np.polyfit(scores, ratios, 1)
    p = np.poly1d(z)

    """crit = p.deriv().r
    r_crit = crit[crit.imag == 0].real
    test = p.deriv(1)(r_crit)

    x_min = r_crit[test > 0]
    y_min = p(x_min)"""

    # plt.plot(x_min, y_min, "o")
    plt.plot(x_plot, p(x_plot), "-")
    plt.plot(scores, ratios, ".")
    plt.xlim(0, max(scores))
    plt.ylim(0, max(ratios))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(scores, windows, zs=back_windows)
    plt.show()


    """tck, u = interpolate.splprep([scores, ratios], s=3)
    unew = np.linspace(0, max(scores)*2, 500)
    out = interpolate.splev(unew, tck)
    plt.figure()
    plt.plot(scores, ratios, ".", out[0], out[1], "-")
    plt.show()"""


def price_distribution(plot=True):
    print("PRICE DISTRIBUTION")
    # plot price distributions, choose threshold based on that distribution
    # ...so that majority class is as small as possible
    ids = np.array(kwargs["IDs"])
    type = kwargs["type"]
    window = kwargs["window"]
    final_set_f = kwargs["final_set_f"]

    ids = final_set_f(None, None, ids)[0]

    price_changes = []
    if type == "articles":
        price_changes, _ = news.get_relative_Y_changes(ids, window)
    elif type == "conversations":
        price_changes, _ = trollbox.get_relative_Y_changes(ids, window)
    elif type == "tweets":
        price_changes, curr = twitter.get_relative_Y_changes(ids, window)
        from collections import Counter
        print(dict(Counter(curr)))
        print(len(curr))

    if price_changes is None or len(price_changes) == 0:
        exit()

    for i, price in enumerate(price_changes):
        price_changes[i] = abs(price)
    price_changes.sort()
    params = stats.lognorm.fit(price_changes, 2)
    # exit()

    # res = Poisson(price_changes, np.ones_like(price_changes)).fit()
    # print(res.summary())
    # mean = res.predict()
    # print(mean)
    # dist = stats.poisson(mean[0])

    if plot:
        weights = np.ones_like(price_changes) / len(price_changes)

        plt.hist(price_changes, bins=1000, label="Porazdelitev cen", normed=True)
        plt.plot(price_changes, stats.lognorm.pdf(price_changes, *params), "-", label="Logaritemsko normalna porazdelitev")

        plt.xlabel("relativna spremmeba cene")
        plt.legend()
        plt.show()

    # this works only if price distribution is cauchy:
    # threshold is determined so that sample data will be split in thirds.
    # https://i.stack.imgur.com/4o1Ex.png
    threshold1 = stats.lognorm.ppf(1/3, *params)

    f = open("results/price_distribution.txt", "a")
    f.write(type + ", window: " + str(window) + ", thirds margin: " + str(threshold1) + "\n")
    f.close()

    print(threshold1)

    return threshold1

# plot_majority_class_corr()
plot_window_corr()
