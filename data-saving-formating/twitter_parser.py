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
    if tweets != None:
        for i in range(len(tweets)):
            d = tweets[i]["message"]["postedTime"].split("T")[0]
            if datetime(2015, 11, 1) <= datetime.strptime(d, "%Y-%m-%d") < datetime(2016, 11, 1):
                actor = tweets[i]["message"]["actor"]
                user_id = "user_" + actor.pop("id", None).replace(":", "=", 1)
                if user_id not in users:
                    actor.pop("summary", None)
                    actor.pop("image", None)
                    actor.pop("utfOffset", None)
                    actor.pop("languages", None)
                    actor.pop("prefferedUsername", None)
                    actor.pop("link", None)
                    actor.pop("twitterTimeZone", None)
                    actor.pop("links", None)
                    actor["_id"] = user_id

                    db.users.insert_one(actor)
                    users.add(user_id)

                twitter_entities = tweets[i]["message"]["twitter_entities"]

                object_ = tweets[i]["message"]["object"]
                object_.pop("link", None)

                favourites_count = tweets[i]["message"]["favoritesCount"]

                retweet_count = tweets[i]["message"]["retweetCount"]

                tweet_id = currency + ":" + object_.pop("id", None).replace("search.twitter.com,", "")

                tweet_data = dict()
                tweet_data["user_id"] = user_id
                tweet_data["favourites_count"] = favourites_count
                tweet_data["retweetCount"] = retweet_count
                tweet_data["_id"] = tweet_id

                for key, value in twitter_entities.items():
                    tweet_data[key] = value
                for key, value in object_.items():
                    if key == "summary":
                        key = "body"
                    tweet_data[key] = value
                try:
                    db.messages.insert_one(tweet_data)
                except pymongo.errors.DuplicateKeyError:
                    print("skipped duplicate message")

                print("saved message: ", tweet_id, ", filename: ", currency + "." + part + ".json", sep="")
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
