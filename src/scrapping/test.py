from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup

import requests
import urllib


def save_click(browser, by, name, wait_time=300):
    WebDriverWait(browser, wait_time).until(EC.element_to_be_clickable((by, name))).click()


browser = webdriver.Firefox()
browser.get('https://www.vinted.de/vetements?catalog[]=12&order=newest_first')


# Get rid of the Cookies - Popup
WebDriverWait(browser, 300).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()

# Wait for the items to be clickable
WebDriverWait(browser, 300).until(EC.element_to_be_clickable((By.CLASS_NAME, "ItemBox_overlay__1kNfX")))

items_on_page = browser.find_elements_by_class_name("ItemBox_overlay__1kNfX")

count = 0
for item in items_on_page:
    url_site = item.get_attribute("href")
    req = requests.get(url_site)
    soup = BeautifulSoup(req.content, "html.parser")
    gallery = soup.find_all("a", class_="item-thumbnail")
    if len(gallery)>1:
        for picture in gallery:
            url = picture.attrs["href"]
            print(url)
            urllib.request.urlretrieve(url, "test_"+str(count)+".jpg")
            count = count + 1


# for item in items_on_page:
#     time.sleep(2) # Wait for popups to move out of the way
#     item.click() # Choose item
#
#     WebDriverWait(browser, 300).until(EC.presence_of_element_located((By.CSS_SELECTOR, "img.item-thumbnail")))
#     gallery_elements = browser.find_elements_by_css_selector("img.item-thumbnail") # Click on one photo of the gallery
#     print(len(gallery_elements), gallery_elements)
#     if len(gallery_elements) > 1:
#         title = gallery_elements[0].get_attribute("title")
#         print(f"Found item with title: {title}")
#
#         gallery_elements[0].click()
#
#         unique_urls = []
#         has_new_urls = True
#         while has_new_urls:
#             print("Iterate_over...")
#             WebDriverWait(browser, 300).until(EC.element_to_be_clickable((By.CLASS_NAME, "fancybox-next"))).click()
#             browser.find_elements_by_class_name("fancybox-next")
#             WebDriverWait(browser, 300).until(EC.presence_of_element_located((By.CLASS_NAME, "fancybox-image")))
#             img = browser.find_element_by_class_name("fancybox-image")
#             url = img.get_attribute("src")
#             if url not in unique_urls:
#                 unique_urls.append(url)
#             else:
#                 has_new_urls = False
#
#         print(unique_urls)
#         # download_file = urllib.request.urlopen()
#         # for i, url in enumerate(unique_urls):
#         #     download_file.retrieve(url, "test_"+str(i)+".jpg")
#
#     browser.back()

