import requests
import codecs
from urllib.parse import quote
from datetime import date
from dateutil.relativedelta import relativedelta
import json
import os

#username = "cb97fa69-ce16-4d6e-9289-a4d6e96183ab"
#password = "r7Z0l49hpE"
username = "bd796191-44ad-42c1-9b2f-5b4b9bbdd357"
password = "SZcWTJ7GdX"
host = "cdeservice.mybluemix.net"
port = "443"


def writeFile(filename, number, response):
    f = codecs.open(filename="data-hashtags/" + filename + "." + str(number).zfill(3) + ".json", mode="w", encoding="utf-8")
    f.write(response.text)
    f.close()


def makeURL(keywords, date_start, date_end):
    url = "https://" + username + ":" + password + \
        "@" + host + ":" + port + "/api/v1/messages/"
    words = "(#" + keywords[0].lower() + " OR #" + keywords[1].lower().replace(" ", "") + ")"
    query = quote("(posted:" + date_start + "," +
                  date_end + " AND " + words + ")")
    search = "search?q=" + query + "&size=3000"
    return url + search


def getCurrencies():
    url = "https://poloniex.com/public?command=returnCurrencies"
    res = requests.get(url)
    json_dict = json.loads(res.text)

    ret = dict()

    for key, values in json_dict.items():
        ret[key] = values["name"]
    return ret

currencies = getCurrencies()
print("all currencies:", len(currencies))

for file in os.listdir("data-hashtags/"):
    currency = file.split(".")[0]
    if currency in currencies:
        del currencies[currency]

#dolla sign exceptions
todo = []
exceptions = ["LOVE", "BTS"]
for exception in exceptions:
    if exception in currencies and exception not in todo:
        del currencies[exception]

if "C2" in currencies:
    currencies["C2"] = "Coin2"

for key, value in currencies.items():
    print(key, value)
print("currencies left:", len(currencies))

i = 0
for curr_key, curr_value in currencies.items():
    from_date = date(2015, 10, 31)
    to_date = date(2016, 10, 31)
    while from_date < to_date:
        fd = str(from_date.year) + "-" + str(from_date.month).zfill(2) + "-" + str(from_date.day).zfill(2)
        from_date = from_date + relativedelta(weeks=+1)
        if from_date > to_date:
            from_date = to_date
        td = from_date + relativedelta(days=-1)
        td = str(td.year) + "-" + str(td.month).zfill(2) + "-" + str(td.day).zfill(2)

        url = makeURL((curr_key, curr_value), fd, td)

        res = requests.get(url)
        json_d = json.loads(res.text)
        print(curr_key, json_d["search"], sep=" : ")

        if json_d["search"]["results"] < 50000:
            writeFile(curr_key, i, res)
            i += 1

            while json_d["search"]["current"] > 0:
                if not os.path.isfile("data-hashtags/" + curr_key + "." + str(i).zfill(3) + ".json"):
                    href = "https://" + username + ":" + password + "@" + json_d["related"]["next"]["href"][8:]
                    res = requests.get(href)
                    json_d = json.loads(res.text)
                    print(i, curr_key, href)

                    if json_d["search"]["current"] > 0:
                        writeFile(curr_key, i, res)
                        i += 1
                    else:
                        #print(json_d)
                        break
                else:
                    print("partially skipped " + curr_key)
        else:
            sk = open(curr_key + "_skipped, " + str(fd), "w")
            sk.write(str(fd) + " : " + str(td))
            sk.close()
            print("Skipped:", curr_key)

    i = 0

