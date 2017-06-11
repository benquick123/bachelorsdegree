import codecs
import json
import os

def writeLinksToFile(currency, json_tree, f_out):
    data = json_tree["data"]["children"]
    for element in data:
        is_post_self = element["data"]["is_self"]
        if not is_post_self:
            f_out.write(element["data"]["url"] + "\n")

last_curr = "CORG"
loading = False

prev_curr = last_curr
f_out = codecs.open("D:/Diploma_data/reddit/news-links/" + prev_curr + ".txt", mode='w')

for filename in os.listdir("D:/Diploma_data/reddit/reddit-data/"):
    currency = filename.split(".")[0]

    if currency == last_curr:
        loading = True

    if loading:
        print(filename)
        if currency != prev_curr:
            f_out.close()
            f_out = open("D:/Diploma_data/reddit/news-links/" + currency + ".txt", mode='w', encoding="utf-8")
            prev_curr = currency

        f_in = codecs.open("D:/Diploma_data/reddit/reddit-data/" + filename, "r", encoding="cp1250", errors="ignore")
        json_tree = json.loads(f_in.read(), encoding="cp1250")
        f_in.close()

        writeLinksToFile(currency, json_tree, f_out)

f_out.close()
