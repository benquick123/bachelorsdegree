import requests
import requests.auth
from urllib.parse import quote
import json
import time
from datetime import datetime

def get_credentials():
	credentials = open("bojdomir.reddit.credentials", "r")
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

def fetch_json(currency, date_from, date_to, offset, count=100):
	global auth_token
	#https://www.reddit.com/r/psdtwk/wiki/cloudsearch#wiki_using_boolean_operators_in_amazon_cloudsearch_text_searches
	headers = dict()
	headers["User-Agent"] = "Cryptocurrency analysis/0.1 by " + "Bojdomir"
	headers["Authorization"] = auth_token["token_type"] + " " + auth_token["access_token"]
	query = quote("(and title:'" + currency + "' is_self:false)")
	url = "https://oauth.reddit.com/search?q=" + query + "&sort=new&t=all&raw_json=1"
	print(url)

	response = requests.get(url, headers=headers)
	response_json = response.json()

	if "error" in response_json:
		print("generating new auth_token...")
		auth_token = get_credentials()
		response_json = fetch_json()

	return response_json

def get_currencies():
    url = "https://poloniex.com/public?command=returnCurrencies"
    res = requests.get(url)
    json_dict = json.loads(res.text)

    currencies = dict()

    for key, values in json_dict.items():
        currencies[key] = values["name"]
    return currencies

def write_file(response, currency, date_from, offset):
	filename = currency + "." + str(date_from) + "." + str(offset/100).zfill(3) + ".json"
	f = open(filename, "w")
	json.dump(response, f)
	f.close()
	print (filename + " saved to disk.")

currencies = get_currencies()
auth_token = get_credentials()

date_end = time.mktime(datetime(2016, 11, 1).timetuple())

for currency in currencies.values():
	date_from = time.mktime(datetime(2015, 10, 31).timetuple())
	date_to = date_from + (60*60*24*7) # +1 week

	while date_from < date_end:
		offset = 0
		while True:
			response_json = fetch_json(currency, date_from, date_to, offset)
			write_file(response_json, currency, date_from, offset)
			offset += 100

			if len(response_json["data"]["children"]) == 0:
				break
		date_from = date_to
		date_to = date_from + (60*60*24*7)