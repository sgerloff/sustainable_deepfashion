from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup

import argparse, os, requests, time, urllib

from src.utility import get_project_dir


class VintedScraper:
    def __init__(self, start_url, category_name="None"):
        self.browser = self.setup_browser()
        self.browser.get(start_url)
        self.accept_cookies()

        self.category_name = category_name
        self.number_of_items = 0

    @staticmethod
    def setup_browser():
        profile = webdriver.FirefoxProfile()
        profile.add_extension(
            "/home/sascha/.mozilla/firefox/v45oij4p.default-release/extensions/adblockultimate@adblockultimate.net.xpi")
        return webdriver.Firefox(profile)

    def accept_cookies(self):
        time.sleep(5)
        self.patient_click(By.ID, "onetrust-accept-btn-handler")

    def patient_click(self, by, name, wait_time=300):
        WebDriverWait(self.browser, wait_time).until(EC.element_to_be_clickable((by, name))).click()

    def crawl(self):
        self.number_of_items = 0
        while True:
            print("Start crawling next page:")
            self.crawl_page()
            WebDriverWait(self.browser, 300).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-testid="next-page"]')))
            next_url = self.browser.find_element_by_css_selector('a[data-testid="next-page"]').get_attribute("href")
            self.browser.get(next_url)

    def crawl_page(self):
        items_on_page = self.get_items_on_page()
        for item in items_on_page:
            self.download_item(item)

    def get_items_on_page(self):
        WebDriverWait(self.browser, 300).until(EC.element_to_be_clickable((By.CLASS_NAME, "ItemBox_overlay__1kNfX")))
        return self.browser.find_elements_by_class_name("ItemBox_overlay__1kNfX")

    def download_item(self, item):
        url = item.get_attribute("href")
        soup = self.get_soup_from_url(item.get_attribute("href"))

        gallery = soup.find_all("a", class_="item-thumbnail")
        if len(gallery) > 1:
            self.download_pictures_of_gallery(gallery)
            self.number_of_items += 1
            self.write_html(soup)
        else:
            print(f"Dismiss item '{url}' with only one picture in gallery.")

    @staticmethod
    def get_soup_from_url(url):
        request_item = requests.get(url)
        return BeautifulSoup(request_item.content, "html.parser")

    def download_pictures_of_gallery(self, gallery):
        for i, pic in enumerate(gallery):
            url = pic.attrs["href"]
            filepath = self.get_file_path(i)
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded '{url}' to {filepath}")

    def get_file_path(self, i):
        filepath = os.path.join(get_project_dir(),
                                "data",
                                "raw",
                                f"vinted_{self.category_name}",
                                f"item_{self.number_of_items:06d}",
                                f"{i:02d}.jpg"
                                )
        dir_name = os.path.dirname(filepath)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        return filepath

    def write_html(self, soup):
        path = self.get_file_path(-1)
        path = os.path.join(os.path.dirname(path), "soup.html")
        with open(path, "w") as file:
            file.write(str(soup))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--url', type=str,
                        help='URL to category of vinted that should be scraped')
    parser.add_argument('--name', type=str, help='name of the category, which is used in naming the folders')

    args = parser.parse_args()

    vinted_scraper = VintedScraper(args.url, category_name=args.name)
    vinted_scraper.crawl()

