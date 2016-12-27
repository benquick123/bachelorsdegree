import codecs
import os
import pymongo
import newspaper

def db_save(db, url, filename):
    a = newspaper.Article(url)
    a.download()
    a.parse()
    a.nlp()
    print(a.title)
    print(a.text)
    print(a.publish_date)
    print(a.keywords)


client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["news"]

for filename in os.listdir("D:/Diploma_data/reddit/news-links/"):
    print(filename)
    file = codecs.open("D:/Diploma_data/reddit/news-links/" + filename, "r", encoding="utf-8", errors="ignore")
    links = file.readlines()
    for link in links:
        link = link.replace("\r\n", "")
        db_save(db, link, filename)
        print(link)
    file.close()
