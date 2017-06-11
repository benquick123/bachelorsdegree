import codecs
import os
import pymongo
import newspaper
from urllib.parse import urlparse


def db_save(db, url, filename, authors, sources):
    a = newspaper.Article(url)

    blacklist = set(["xhamster.com", "i.imgur.com", "youtu.be", "www.ebay.com.au", "imgur.com", "www.youtube.com", "www.freebdsmtrailer.com", "gfycat.com", "soundcloud.com",
                    "spankbang.com", "i.redd.it", "www.instagram.com", "twitter.com", "vid.me", "www.gfycat.com", "www.alphaporno.com", "www.reddit.com", "www.twitter.com", "i.reddituploads.com"])
    blacklist_formats = set([".jpg", ".png", ".mp3", ".mp4", ".gif", ".svg", ".exe", ".pdf", ".zip", ".webm", ".m4r", ".bz2"])
    source = urlparse(url).hostname
    ending = "." + url.split(".")[-1]

    if source not in blacklist and ending not in blacklist_formats:
        try:
            print("WORKING ON", url)
        except UnicodeEncodeError:
            print("WORKING ON", source)
        a.download()
        if a.is_downloaded:
            try:
                a.parse()
                while not a.is_parsed:
                    pass

                if a.publish_date is not None:
                    a.nlp()

                    article_authors = a.authors
                    if len(article_authors) == 0:
                        article_authors = ["unknown"]

                    for _author in article_authors:
                        if _author not in authors:
                            values = dict()
                            values["_id"] = _author
                            values["credibility"] = -1
                            db.authors.insert_one(values)
                            authors.add(_author)

                    if source not in sources:
                        values = dict()
                        values["_id"] = source
                        values["credibility"] = -1
                        db.sources.insert_one(values)
                        sources.add(source)

                    currency = filename.split(".")[0]
                    news_values = {
                        "title": a.title,
                        "text": a.text,
                        "date": a.publish_date,
                        "summary": a.summary,
                        "authors": a.authors,
                        "keywords": a.keywords,
                        "meta_keywords": a.meta_keywords,
                        "additional": a.additional_data,
                        "tags": list(a.tags),
                        "meta_description": a.meta_description,
                        "source": source,
                        "subjectivity": -1,
                        "sentiment": -1,
                        "_id": currency + ":" + a.publish_date.strftime("%Y-%m-%d-%H-%M") + ":" + a.title.replace(" ", "").lower()
                    }

                    try:
                        db.articles.insert_one(news_values)
                        print("saved", source, filename)
                    except pymongo.errors.DuplicateKeyError:
                        print("skipped duplicate", source)
            except UnicodeEncodeError:
                print("UnicodeEncodeError", source)


    else:
        print("skipped " + source)

    return authors, sources

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db = client["news"]

authors = set()
sources = set()

for author in db.authors.find():
    authors.add(author["_id"])

for source in db.sources.find():
    sources.add(source["_id"])

last = "http://ij.org/cops-seized-over-107000-from-couple-didnt-charge-them-with-a-crime/"
loading = False

for filename in os.listdir("D:/Diploma_data/reddit/news-links/"):
    print(filename)
    file = codecs.open("D:/Diploma_data/reddit/news-links/" + filename, "r", encoding="utf-8", errors="ignore")
    links = file.readlines()
    for link in links:
        link = link.replace("\r\n", "")
        if link == last:
            loading = True
        if loading:
            authors, sources = db_save(db, link, filename, authors, sources)

    file.close()
