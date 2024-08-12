import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from json_utils import save_json
from newspaper import Article


class SearchHandler:
    def __init__(self):
        load_dotenv()
        self._api_key = os.getenv("GOOGLE_API_KEY")
        self._engine_id = os.getenv("GOOGLE_ENGINE_ID")
        if not self._api_key or not self._engine_id:
            raise ValueError(
                "API key or Engine ID is missing from environment variables"
            )

    def get_urls(self, query, num=10):
        """
        Returns the top 'num' search results from Google Custom Search API
        Note: num is capped at 10 due to the API limitations.
        """
        try:
            num = min(num, 10)
            url = f"https://www.googleapis.com/customsearch/v1?key={self._api_key}&cx={self._engine_id}&q={query}&num={num}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return [
                {"title": item["title"], "link": item["link"]}
                for item in data["items"]
            ]
        except requests.exceptions.RequestException as e:
            print(f"HTTP request error: {e}")
            try:
                print(response.json())
            except:
                print("No JSON response received")

            return []
        except KeyError as e:
            print(f"Error parsing response data: {e}")
            return []

    @staticmethod
    def parse_source_from_url(url: str):
        """
        Parses the source from the url like 'https://www.example.com/article' to 'example'
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        source = domain.split(".")[1]
        return source

    @staticmethod
    def parse_article(
        url: str, rank: int, max_retries: int = 3, backoff_factor: float = 0.5
    ):
        """
        Parses the article text from the url using newspaper3k, with retry mechanism for handling connection errors.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                article = Article(url)
                article.download()
                article.parse()
                return {
                    "rank": rank,
                    "title": article.title,
                    "source": SearchHandler.parse_source_from_url(url),
                    "url": url,
                    "text": article.text,
                }
            except requests.exceptions.ConnectionError as e:
                print(
                    f"Connection error: {e}. Retrying in {backoff_factor * (2 ** attempt)} seconds..."
                )
                time.sleep(backoff_factor * (2**attempt))
                attempt += 1
            except Exception as e:
                print(
                    f"Error parsing article: {e} Retrying in {backoff_factor * (2 ** attempt)} seconds..."
                )
                time.sleep(backoff_factor * (2**attempt))
                attempt += 1
                break
        return None

    def search(self, query, num=10):
        num = min(num, 10)
        urls = self.get_urls(query, num)
        results = []
        for url in urls:
            article = self.parse_article(url["link"], urls.index(url) + 1)
            if article:
                results.append(article)
        return {"question": query, "top_k": num, "results": results}


if __name__ == "__main__":
    search_handler = SearchHandler()
    results = search_handler.search("who is the president of usa 2024", num=10)
    os.makedirs("../data/search", exist_ok=True)
    save_json("../data/search/search_tests.json", results)
