import os
import numpy as np
import matplotlib.pyplot as plt


def blankspace_fix(filename):
    f = open("results/parameter_search/" + filename, "r")
    s = f.read()
    f.close()
    i = s.find("]") + 1
    s = s[:i] + ", " + s[i:]
    f = open("results/parameter_search/" + filename, "w")
    f.write(s)
    f.close()


def read_file(filename):
    params = dict()
    f = open("results/parameter_search/" + filename, "r")
    s = f.read()
    f.close()
    s = ", ".join(s.split("\n")[:-2]).split(", ")
    for _s in s:
        _s = _s.replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace(":", "").split(" ")
        # print(_s)
        if _s[0] == "1" or _s[0] == "-1" or len(_s) == 3:
            _d = {int(_s[-2]): int(_s[-1])}
            if "classes" not in params:
                params["classes"] = dict()
            params["classes"][int(_s[-2])] = int(_s[-1])
        elif _s[0] == "back_windows":
            params[_s[0]] = [int(_s[1])]
        elif len(_s) == 1:
            params["back_windows"].append(int(_s[0]))
        else:
            params[_s[0]] = float(_s[-1]) if float(_s[-1]) % 1 != 0 else int(_s[-1])

    # print(params)

    return params


def converging_average(all_params):
    scores = []
    avgs = []
    for params in all_params:
        classes = params["classes"]
        major = max(classes.values()) / sum(classes.values())
        scores.append(params["score"] - major)
        avgs.append(np.mean(scores))

    plt.plot(range(len(avgs)), avgs, "-")
    plt.show()


def find_best(all_params):
    best_i = -1
    best_score = -1
    for i, params in enumerate(all_params):
        classes = params["classes"]
        major = max(classes.values()) / sum(classes.values())
        if best_score < params["score"] - major:
            best_score = params["score"] - major
            best_i = i

    print(best_score)
    return best_i


def parse_files():
    params = []
    for filename in os.listdir("results/parameter_search"):
        _params = read_file(filename)
        params.append(_params)
        # blankspace_fix(filename)

    # np.random.shuffle(params)
    converging_average(params)

    best = find_best(params)
    print(params[best])

os.chdir("..")
parse_files()
