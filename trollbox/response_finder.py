import pymongo

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["trollbox"]


# users_responses = dict()
users_last_message = dict()
users = dict()

for entry in db.users.find():
    users_last_message[entry["_id"]] = None
    users[entry["_id"]] = entry["user_name"]
    users[entry["user_name"]] = entry["_id"]


for entry in db.messages.find().sort("_id", 1):
    users_last_message[entry["user_id"]] = entry["_id"]

    first_word = entry["text"].split(",")[0]
    if first_word in users:
        if users_last_message[users[first_word]] is not None:
            if abs(users_last_message[users[first_word]] - entry["_id"]) < 1000:
                db.messages.update_one({"_id": users_last_message[users[first_word]]}, {"$push": {"responses": entry["_id"]}})
                print("UPDATE:", "message", users_last_message[users[first_word]], "updated with response", entry["_id"])
            else:
                users_last_message[users[first_word]] = None
                print("ERROR:", first_word, "is out of reach.")
        else:
            try:
                print("ERROR:", first_word + " (" + str(users_last_message[users[first_word]]) + ")", "is out of reach or has no previous messages:", entry["text"])
            except UnicodeEncodeError:
                print("UnicodeEncodeError: not harmful")
    else:
        pass
