import pymongo

def build_tree(_id, db, print=True):
    tree = db.messages.find_one({"_id": _id})
    responses = tree["responses"]
    tree["responses"] = []
    for response_id in responses:
        to_append = build_tree(response_id, db, print=False)
        tree["responses"].append(to_append)

    if print == True:
        print_tree(tree)
    return tree


def print_tree(tree, level=0):
    for i in range(level):
        if i < level-1:
            print("    ", end="")
        else:
            print("|___", end="")

    print(str(tree["_id"]) + ": " + tree["text"] + "\n", end="")

    responses = tree["responses"]
    for response in responses:
        print_tree(response, level+1)

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["trollbox"]

build_tree(3881513, db)
