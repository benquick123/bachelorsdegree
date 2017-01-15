import pymongo
import codecs
import json
import os
from datetime import datetime


def save_to_db(db, users, currency, part):
    tweets_file = codecs.open("D:/Diploma_data/twitter-data/twitter-data/" + currency + "." + part + ".json", encoding="cp1250", errors="ignore")
    tweets = json.loads(tweets_file.read())
    tweets_file.close()

    tweets = tweets["tweets"]
    if tweets != None and len(tweets) > 0:
        print(currency, part)
        for i in range(len(tweets)):
            tweet = tweets[i]
            d = tweet["message"]["postedTime"].split("T")[0]
            if datetime(2015, 11, 1) <= datetime.strptime(d, "%Y-%m-%d") < datetime(2016, 11, 1):
                actor = tweet["message"]["actor"]
                actor_id = actor["id"].split(":")[-1]
                if actor_id not in users:
                    try:
                        actor_type = actor["person"]
                    except KeyError:
                        actor_type = ""

                    actor_data = {
                        "_id": actor_id,
                        "verified": actor["verified"],
                        "friends_count": actor["friendsCount"],
                        "favorites_count": actor["favoritesCount"],
                        "person": actor_type,
                        "followers_count": actor["followersCount"],
                        "credibility": -1
                    }

                    db.users.insert_one(actor_data)
                    users.add(actor_data["_id"])

                try:
                    _sentiment_evidence = tweet["cde"]["content"]["sentiment"]["evidence"]
                    _sentiment = tweet["cde"]["contnet"]["sentiment"]["polarity"]
                except KeyError:
                    _sentiment_evidence = []
                    _sentiment = ""

                tweet_data = {
                    "_sentiment_evidence": _sentiment_evidence,
                    "_sentiment": _sentiment,
                    "sentiment": -1,
                    "subjectivity": -1,
                    "_id": tweet["message"]["id"].split(":")[-1],
                    "posted_time": tweet["message"]["postedTime"],
                    "text": tweet["message"]["body"],
                    "favorites_count": tweet["message"]["favoritesCount"],
                    "author": tweet["message"]["actor"]["id"].split(":")[-1],
                    "retweet_count": tweet["message"]["retweetCount"],
                    "crypto_currency": currency
                }

                try:
                    db.messages.insert_one(tweet_data)
                except pymongo.errors.DuplicateKeyError:
                    print("skipped duplicate message")

                print("saved message: ", tweets[i]["message"]["object"]["id"], ", filename: ", currency + "." + part + ".json", sep="")
            else:
                print("skipped message: ", tweets[i]["message"]["object"]["id"], ", filename: ", currency + "." + part + ".json", sep="")


client = pymongo.MongoClient(host="localhost", port=27017)
db = client.twitter

s = set()
k = db.messages.distinct("_id")
#k = r.keys("*")
for key in k:
	s.add(key.split(":")[0])

print(len(s))
last_currency = "1CR"
last_part = 0

users = set(db.users.distinct("_id"))

for file in os.listdir("D:/Diploma_data/twitter-data/twitter-data/"):
    file_split = file.split(".")
    currency = file_split[0]
    part = file_split[1]
    if (currency not in s) or (currency == last_currency and int(part) > last_part):
    	save_to_db(db, users, currency, part)

# message
# favouritesCount,
# object[
#	summary, postedTime, id, objectType],
# retweetCount,
# twitter_entities[
#	urls, hashtags, user_mentions, symbols],
# actor[
# statusesCount, displayName, postedTime, verified, friendsCount,
# favouritesCount, objectType, id, followerCount]
