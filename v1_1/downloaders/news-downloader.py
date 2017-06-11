import requests
import requests.auth
from urllib.parse import quote
import json
import time
from datetime import datetime
import os


def get_credentials():
    credentials = open("../credentials/bojdomir.reddit.credentials", "r")
    username = credentials.readline().replace("\n", "")
    password = credentials.readline().replace("\n", "")
    app_id = credentials.readline().replace("\n", "")
    secret = credentials.readline().replace("\n", "")

    client_auth = requests.auth.HTTPBasicAuth(app_id, secret)
    post_data = {"grant_type": "password", "username": username, "password": password, "duration":"permanent"}
    headers = {"User-Agent": "Cryptocurrency analysis/0.1 by " + username}

    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
    token = response.json()
    return token


def fetch_json(currency, date_from, date_to, after=None, count=100):
    global auth_token
    #https://www.reddit.com/r/psdtwk/wiki/cloudsearch#wiki_using_boolean_operators_in_amazon_cloudsearch_text_searches
    headers = dict()
    headers["User-Agent"] = "Cryptocurrency analysis/0.1 by " + "Bojdomir"
    headers["Authorization"] = auth_token["token_type"] + " " + auth_token["access_token"]
    query = quote("(and timestamp:" + str(date_from) + ".." + str(date_to) + " title:'" + currency + "')")
    url = "https://oauth.reddit.com/search?q=" + query + "&sort=new&t=all&limit=" + str(count) + "&raw_json=1&syntax=cloudsearch"
    if after != None:
        url = url + "&after=" + after
    print(url)

    response = requests.get(url, headers=headers)
    response_json = response.json()

    if "error" in response_json:
        print("generating new auth_token...")
        auth_token = get_credentials()
        response_json = fetch_json(currency, date_from, date_to, after=after)

    return response_json


def get_currencies():
    url = "https://poloniex.com/public?command=returnCurrencies"
    res = requests.get(url)
    json_dict = json.loads(res.text)

    currencies = dict()

    for key, values in json_dict.items():
        currencies[key] = values["name"]
    return currencies


def write_file(response, currency, date_from, after):
    filename = currency + "." + str(date_from) + "." + str(after) + ".json"
    f = open("news-data/" + filename, "w")
    json.dump(response, f)
    f.close()

currencies = get_currencies()
auth_token = get_credentials()

date_end = time.mktime(datetime(2016, 11, 1).timetuple())

s = set()
for file in os.listdir("news-data/"):
    s.add(file.split(".")[0])

for key, currency in currencies.items():
    if key not in s:
        date_from = time.mktime(datetime(2015, 10, 31).timetuple())
        date_to = date_from + (60*60*24*7) # +1 week

        while date_from < date_end:
            offset = 0
            after = None
            while True:
                a_time = time.time()

                response_json = fetch_json(currency, int(date_from), int(date_to), after)
                after = response_json["data"]["after"]
                print(currency + ": found " + str(len(response_json["data"]["children"])) + " results.")
                write_file(response_json, key, int(date_from), after)

                b_time = time.time()
                diff = b_time - a_time
                if diff < 1:
                    time.sleep(1-diff)

                if after == None:
                    break

            date_from = date_to
            date_to = date_from + (60*60*24*7)
    else:
        print("skipped " + key + ".")
