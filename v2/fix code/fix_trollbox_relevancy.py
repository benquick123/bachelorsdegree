import pymongo
import v1_1.trollbox.v1.conversations as conversations
import v1_1.trollbox.v1.response_tree as response_tree

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db_trollbox = client["trollbox"]
db_prices = client["crypto_prices"]

prices = set(db_prices.collection_names())

# fix the initial screwup
# for conversation in db_trollbox.conversations.find({"responses": True}):
#    tree = response_tree.build_tree(conversation["_id"], db_trollbox, print=False)
#    coin_mentions = conversations.all_mentioned_currencies(tree)
#    db_trollbox.conversations.update_one({"_id": conversation["_id"]}, {"$set": {"coin_mentions": list(coin_mentions)}})

for conversation in db_trollbox.conversations.find():
    coin_mentions = set(conversation["coin_mentions"]).intersection(prices)
    db_trollbox.conversations.update_one({"_id": conversation["_id"]}, {"$set": {"coin_mentions": list(coin_mentions)}})

