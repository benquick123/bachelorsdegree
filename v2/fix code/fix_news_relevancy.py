import pymongo

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db_news = client["news"]
db_prices = client["crypto_prices"]

currencies = set(db_prices.collection_names())

for article in db_news.articles.find({"$or": [{"relevant": True}, {"relevant_predict": True}]}):
    if article["currency"].lower() not in currencies:
        db_news.articles.update_one({"_id": article["_id"]}, {"$set": {"relevant": False}})
    else:
        db_news.articles.update_one({"_id": article["_id"]}, {"$set": {"relevant": True}})
