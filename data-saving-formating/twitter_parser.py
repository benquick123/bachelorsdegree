import redis
import codecs
import json
import os
from datetime import datetime


def save_to_db(r, currency, part):
    tweets_file = codecs.open("D:/Diploma_data_backup/twitter-data/twitter-data/" +
                              currency + "." + part + ".json", encoding="cp1250", errors="ignore")
    tweets = json.loads(tweets_file.read())
    tweets_file.close()

    tweets = tweets["tweets"]
    if tweets != None:
        for i in range(len(tweets)):
            d = tweets[i]["message"]["postedTime"].split("T")[0]
            if datetime(2015, 11, 1) <= datetime.strptime(d, "%Y-%m-%d") < datetime(2016, 11, 1):
                actor = tweets[i]["message"]["actor"]
                user_id = "user_" + actor.pop("id", None).replace(":", "=", 1)
                if len(r.keys(user_id)) == 0:
                    actor.pop("summary", None)
                    actor.pop("image", None)
                    actor.pop("utfOffset", None)
                    actor.pop("languages", None)
                    actor.pop("prefferedUsername", None)
                    actor.pop("link", None)
                    actor.pop("twitterTimeZone", None)
                    actor.pop("links", None)

                    r.hmset(user_id, actor)

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
                for key, value in twitter_entities.items():
                    tweet_data[key] = value
                for key, value in object_.items():
                    tweet_data[key] = value

                r.hmset(tweet_id, tweet_data)

                print("saved message: ", tweet_id, ", filename: ", currency + "." + part + ".json", sep="")
            else:
                print(d)
                print("skipped message: ", tweets[i]["message"]["object"]["id"], ", filename: ", currency + "." + part + ".json", sep="")


r = redis.StrictRedis(host="localhost", port=6379, db=1)

save_to_db(r, "1CR", "000")
save_to_db(r, "AC", "000")
save_to_db(r, "BTC", "301")

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
