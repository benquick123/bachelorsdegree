
def mark_in_event(db_news, db_twitter, events):
    tweets = set()
    articles = set()
    for event in events.values():
        tweets = tweets.union(set(event.tweet_ids))
        articles = articles.union(set(event.article_ids))

    for tweet in db_twitter.messages.find({"$or": [{"relevant_classifier": True}, {"relevant_predict": True}]}):
        if tweet["_id"] in tweets:
            db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"in_event": True}})
        else:
            db_twitter.messages.update_one({"_id": tweet["_id"]}, {"$set": {"in_event": False}})

    for article in db_news.articles.find({"$or": [{"relevant": True}, {"relevant_predict": True}]}):
        if article["_id"] in articles:
            db_news.articles.update_one({"_id": article["_id"]}, {"$set": {"in_event": True}})
        else:
            db_news.articles.update_one({"_id": article["_id"]}, {"$set": {"in_event": False}})
