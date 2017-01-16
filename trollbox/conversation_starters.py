import pymongo

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["trollbox"]

all_messages = set()
to_remove = set()

for entry in db.messages.find():
    print(entry["_id"])
    if len(entry["responses"]) > 0:
        all_messages.add(entry["_id"])
    to_remove.union(set(entry["responses"]))

final = all_messages - to_remove
print(final)
