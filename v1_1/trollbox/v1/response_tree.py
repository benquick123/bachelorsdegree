import pymongo


def build_tree(_id, db, print=True):
    tree = db.messages.find_one({"_id": _id})

    responses = tree["responses"]
    tree["responses"] = []
    for response_id in responses:
        to_append = build_tree(response_id, db, print=False)
        tree["responses"].append(to_append)

    if print:
        print_tree(tree)
    return tree


def print_tree(tree, level=0, multiple=set()):
    for i in range(level):
        if i in multiple and i < level-1:
            print("|   ", end="")
        elif i < level-1:
            print("    ", end="")
        else:
            print("|___", end="")

    print(str(tree["_id"]) + ": " + tree["text"])

    responses = tree["responses"]
    if len(responses) > 1:
        multiple.add(level)

    for i, response in enumerate(responses):
        if len(responses)-1 == i and level in multiple:
            multiple.remove(level)
        print_tree(response, level+1, multiple)

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["trollbox"]

build_tree(6929663, db)
