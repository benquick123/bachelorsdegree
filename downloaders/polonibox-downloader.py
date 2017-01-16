from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import codecs

def next_page(browser):
    try:
        element = WebDriverWait(browser, 10).until(EC.presence_of_element_located((By.LINK_TEXT, ">")))
        element.click()
    except TimeoutError:
        print("Timeout error")

def save_page(browser, page):
    content = browser.page_source

    file_name = "trollbox_data/page" + str(page) + ".html"
    f = codecs.open(file_name, "w", "utf-8")
    f.write(content)
    f.close()

browser = webdriver.Chrome()
browser.get("http://www.polonibox.com/")

input("Waiting")
next_page(browser)

for i in range(65947, 118000):
    save_page(browser, i)
    next_page(browser)

browser.close()




