from lxml import html
import codecs
import pymongo
from datetime import datetime

def save_to_db(db, users, page_name):
    page = codecs.open("D:/Diploma_data/trollbox-data/trollbox-data/" + page_name, encoding="cp1250", errors="replace")
    page_tree = page.read()
    page.close()
    page_tree = html.fromstring(page_tree)

    table_body = page_tree.xpath('//div[@data-linkify="this"]/table/tbody')[0]

    message_text_row = table_body.xpath('tr/td[2]')
    all_texts = [td.xpath('text()') for td in message_text_row]

    message_text = [" " if len(s) == 0 else s[0] for s in all_texts]
    message_id = table_body.xpath('tr/td[1]/small/a/@href')
    reputation = table_body.xpath('tr/td[1]/span[@title="Reputation"]/text()')
    user = table_body.xpath('tr/td[1]/a/b/text()')
    date = table_body.xpath('tr//td[1]//small/span/@title')

    if len(user) == len(reputation) == len(message_text) == len(message_id) == len(date) == 100:
        for i in range(100):
            date_values = date[i].split(" ")[0]
            if datetime(2015, 11, 1) <= datetime.strptime(date_values, "%Y-%m-%d") < datetime(2016, 11, 1):
                if user[i] not in users:
                    values = {"user_name": user[i], "reputation": reputation[i], "credibility": -1}
                    insert = db.users.insert_one(values)
                    users[user[i]] = insert.inserted_id
                    print("adding new user_id:", insert.inserted_id)

                user_id = users[user[i]]

                mess_id = int(message_id[i].replace("?", "").split("=")[-1])
                values = {"date": date[i], "text": message_text[i], "user_id": user_id, "_id": mess_id, "responses": [], "sentiment": -1, "subjectivity": -1, "context": [], "coin_mentions": []}
                try:
                    db.messages.insert_one(values)
                    print("saved page ",  page_name, ", message ", mess_id, sep="")
                except pymongo.errors.DuplicateKeyError:
                    print("skipped duplicate message")
            else:
                print("skipped page ",  page_name, ", message ", message_id[i].replace("?", ""), sep="")


client = pymongo.MongoClient(host="localhost", port=27017)
db = client["trollbox"]

users = dict()
for user in db.users.find():
    users[user["user_name"]] = user["_id"]

#pages missing: 39357
for page_num in range(68201, 79333):
    if page_num != 39357:
        page_name = "page" + str(page_num) + ".html"
        user_id_counter = save_to_db(db, users, page_name)
