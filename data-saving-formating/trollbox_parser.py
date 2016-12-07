from lxml import html
import codecs
import redis
from datetime import datetime

def save_to_db(r, page_name, user_id_counter):
    page = codecs.open("D:/Diploma_data_backup/trollbox-data/trollbox-data/" + page_name, encoding="cp1250", errors="replace")
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
                if user[i] not in all_users:
                    print("adding new user_id:", user_id_counter)
                    all_users[user[i]] = user_id_counter
                    values = {"user_name": user[i], "reputation": reputation[i], "credibility": ""}
                    user_id = "user_id=" + str(user_id_counter)
                    r.hmset(user_id, values)
                    user_id_counter += 1

                user_id = all_users[user[i]]

                mess_id = message_id[i].replace("?", "")
                values = {"date": date[i], "text": message_text[i], "user_id": user_id, "responses": dict(), "sentiment": "", "subjectivity": "", "context": "", "coin_mentions": ""}
                r.hmset(mess_id, values)
                print("saved page ",  page_name, ", message ", message_id[i], sep="")
            else:
                print("skipped page ",  page_name, ", message ", message_id[i], sep="")

    return user_id_counter

r = redis.StrictRedis(host="localhost", port=6379, db=0)

all_users = dict()
all_user_ids = r.keys("user_id=*")
for user_id in all_user_ids:
    all_users[r.hget(user_id, "user_name").decode("utf-8")] = user_id.decode("utf-8")

user_id_counter = len(all_users)

#pages missing: 39357
for page_num in range(69330, 79333):
    if page_num != 39357:
        page_name = "page" + str(page_num) + ".html"
        user_id_counter = save_to_db(r, page_name, user_id_counter)

r.save()
