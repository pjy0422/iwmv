import os
from urllib.parse import urlparse

import nltk
import requests
from dotenv import load_dotenv
from json_utils import save_json
from newspaper import Article


class SearchHandler:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.engine_id = os.getenv("GOOGLE_ENGINE_ID")

    def get_urls(self, query, num=10):
        """
        Args: query: str, num: int
        Returns: list

        Returns the top 'num' search results from Google Custom Search API
        """
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.engine_id}&q={query}&num={num}&start=1"
        response = requests.get(url)
        data = response.json()
        return [
            {"title": item["title"], "link": item["link"]}
            for item in data["items"]
        ]

    @staticmethod
    def parse_source_from_url(url: str):
        """
        Args: url: str
        Returns: str

        Parses the source from the url like 'https://www.example.com/article' to 'example'
        """

        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        source = domain.split(".")[1]
        return source

    def search(self, query, num=10):
        urls = self.get_urls(query, num)
        results = []
        for url in urls:
            article = Article(url["link"])
            article.download()
            article.parse()
            results.append(
                {
                    "rank": urls.index(url) + 1,
                    "title": url["title"],
                    "url": url["link"],
                    "source": self.parse_source_from_url(url["link"]),
                    "text": article.text,
                }
            )
        return {"question": query, "results": results}


if __name__ == "__main__":
    search_handler = SearchHandler()
    results = search_handler.search(
        "who is the president of usa now 2024", num=5
    )
    save_json("search_tests.json", results)
