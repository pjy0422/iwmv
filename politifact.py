import asyncio
import json
import logging
import re

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Configure logging for error handling
logging.basicConfig(
    filename="scraper_errors.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Constants
BASE_LISTING_URL = "https://www.politifact.com/factchecks/list/?page={page}&speaker=instagram-posts"
CONCURRENT_REQUESTS = 5  # Number of concurrent HTTP requests
JSON_FILE = "instagram_posts_unique_sources.json"

# Regex pattern for matching article URLs
ARTICLE_URL_PATTERN = re.compile(
    r"^https://www\.politifact\.com/factchecks/\d{4}/[a-z]{3}/\d{2}/instagram-posts/[^/]+/?$"
)

# Headers for HTTP requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; YourBotName/1.0; +http://yourwebsite.com/bot)"
}


def get_listing_url(base_url, page_number):
    """
    Generate a single listing URL based on the page number.

    Parameters:
        base_url (str): The base URL with a placeholder for the page number.
        page_number (int): The current page number.

    Returns:
        str: The formatted listing URL.
    """
    return base_url.format(page=page_number)


def extract_article_urls(listing_soup):
    """
    Extract article URLs from a listing page's BeautifulSoup object.

    Parameters:
        listing_soup (BeautifulSoup): Parsed HTML of the listing page.

    Returns:
        set: A set of article URLs matching the desired pattern.
    """
    article_urls = set()

    # Find all <a> tags within the listing page
    for a_tag in listing_soup.find_all("a", href=True):
        href = a_tag["href"]
        # Ensure the URL is absolute
        if href.startswith("/"):
            href = "https://www.politifact.com" + href
        elif not href.startswith("http"):
            href = "https://www.politifact.com/" + href.lstrip("/")
        # Match the URL against the pattern
        if ARTICLE_URL_PATTERN.match(href):
            article_urls.add(href)

    return article_urls


def extract_unique_sources(article_soup):
    """
    Extract unique source names from the 'Our Sources' section of an article.

    Parameters:
        article_soup (BeautifulSoup): Parsed HTML of the article page.

    Returns:
        set: A set of unique source names.
    """
    unique_sources = set()
    sources_section = article_soup.find("section", id="sources")
    if sources_section:
        source_paragraphs = sources_section.find_all("p")
        for p in source_paragraphs:
            links = p.find_all("a")
            if links:
                # Extract the text before the first link and the link text
                text_parts = []
                for content in p.contents:
                    if content == links[0]:
                        break
                    if isinstance(content, str):
                        text_parts.append(content.strip())
                text_before_link = " ".join(text_parts)
                source_name = (
                    text_before_link + " " + links[0].text.strip()
                ).strip()
                # Simplify the source name by removing extra details after commas
                source_name_simple = source_name.split(",")[0].strip()
                if source_name_simple:
                    unique_sources.add(source_name_simple)
            else:
                # No links, use the paragraph text as source name
                source_name = p.text.strip()
                source_name_simple = source_name.split(",")[0].strip()
                if source_name_simple:
                    unique_sources.add(source_name_simple)
    else:
        # Log if 'Our Sources' section is not found
        logging.error("No 'Our Sources' section found in the article.")
    return unique_sources


async def fetch(session, url):
    """
    Asynchronously fetch the content of a URL.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session to use.
        url (str): The URL to fetch.

    Returns:
        str: The text content of the response, or None if an error occurs.
    """
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            return await response.text()
    except aiohttp.ClientError as e:
        logging.error(f"Error fetching URL {url}: {e}")
    except asyncio.TimeoutError:
        logging.error(f"Timeout fetching URL {url}")
    return None


async def process_article(session, url, processed_articles):
    """
    Process a single article URL to extract unique sources.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session to use.
        url (str): The article URL.
        processed_articles (set): Set of already processed article URLs.

    Returns:
        dict or None: A dictionary with article URL and sources, or None if failed.
    """
    if url in processed_articles:
        return None

    article_html = await fetch(session, url)
    if not article_html:
        return None

    article_soup = BeautifulSoup(article_html, "html.parser")
    sources = extract_unique_sources(article_soup)

    if sources:
        processed_articles.add(url)
        return {"Article URL": url, "Sources": sorted(sources)}
    else:
        return None


async def process_listing_page(
    session, page_number, semaphore, processed_articles
):
    """
    Process a single listing page to extract and process article URLs.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session to use.
        page_number (int): The current listing page number.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        processed_articles (set): Set of already processed article URLs.

    Returns:
        list: A list of dictionaries containing article URLs and their sources.
        bool: Indicator whether any articles were found on this page.
    """
    listing_url = get_listing_url(BASE_LISTING_URL, page_number)
    listing_html = await fetch(session, listing_url)
    if not listing_html:
        return [], False

    listing_soup = BeautifulSoup(listing_html, "html.parser")
    article_urls = extract_article_urls(listing_soup)

    if not article_urls:
        return [], False

    articles = []

    async def semaphore_process(url):
        async with semaphore:
            return await process_article(session, url, processed_articles)

    tasks = [semaphore_process(url) for url in article_urls]
    for f in tqdm_asyncio.as_completed(
        tasks,
        total=len(tasks),
        desc=f"Processing Articles on Page {page_number}",
    ):
        result = await f
        if result:
            articles.append(result)

    return articles, True


async def main():
    processed_articles = set()
    all_articles_sources = []
    page_number = 1
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    connector = aiohttp.TCPConnector(limit_per_host=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector
    ) as session:
        while True:
            print(f"\nProcessing listing page {page_number}...")
            articles, has_more = await process_listing_page(
                session, page_number, semaphore, processed_articles
            )
            if not has_more:
                print(f"No articles found on page {page_number}. Stopping.")
                break
            if articles:
                all_articles_sources.extend(articles)
                print(
                    f"  Extracted {len(articles)} unique sources from page {page_number}."
                )
            else:
                print(
                    f"  No sources extracted from articles on page {page_number}."
                )

            page_number += 1

    # Save the collected data to a JSON file asynchronously
    try:
        async with aiofiles.open(JSON_FILE, "w", encoding="utf-8") as f:
            await f.write(
                json.dumps(all_articles_sources, indent=4, ensure_ascii=False)
            )
        print(f"\nData has been saved to {JSON_FILE}")
    except IOError as e:
        logging.error(f"Error writing to JSON file {JSON_FILE}: {e}")
        print(f"Error writing to JSON file: {e}")


if __name__ == "__main__":
    asyncio.run(main())
