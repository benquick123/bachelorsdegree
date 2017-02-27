import pymongo
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

client = pymongo.MongoClient(host="127.0.0.1", port=27017)
db_news = client["news"]

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

last = "SRG:2016-09-01-00-00:couldmonero'spricesurgebetiedtothebitfinexhack?"
cont = False

for article in db_news.articles.find({}, {"_id": 1, "title": 1, "text": 1}).sort("_id", 1):
    if article["_id"] == last:
        cont = True

    if cont:
        _id = article["_id"]
        title = article["title"]
        text = article["text"]

        text = tokenizer.tokenize(text)
        title = tokenizer.tokenize(title)

        text = [stemmer.stem(word.lower()) for word in text if word not in stopwords.words("english")]
        title = [stemmer.stem(word.lower()) for word in title if word not in stopwords.words("english")]

        db_news.articles.update({"_id": _id}, {"$set": {"stemmed_text": text, "stemmed_title": title}})
        print("UPDATE:", _id)
