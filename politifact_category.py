import asyncio
import json
import logging
import re
import argparse
import sys

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

# Configure logging for error handling
logging.basicConfig(
    filename="scraper_errors.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Constants
BASE_LISTING_URL = "https://www.politifact.com/factchecks/list/?page={page}&category={category}"
CONCURRENT_REQUESTS = 5  # Number of concurrent HTTP requests

# Headers for HTTP requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; YourBotName/1.0; +http://yourwebsite.com/bot)"
}

# List of non-English editions to exclude
NON_ENGLISH_EDITIONS = [
    "PolitiFact en Español",
    "PolitiFact en français",
    "PolitiFact en Deutsch",
    # Add more non-English editions as needed
]


def get_listing_url(base_url, page_number, category):
    """
    Generate a single listing URL based on the page number and category.

    Parameters:
        base_url (str): The base URL with placeholders for the page number and category.
        page_number (int): The current page number.
        category (str): The category to scrape.

    Returns:
        str: The formatted listing URL.
    """
    return base_url.format(page=page_number, category=category)


def extract_article_urls(listing_soup):
    """
    Extract article URLs and speakers from a listing page's BeautifulSoup object.

    Parameters:
        listing_soup (BeautifulSoup): Parsed HTML of the listing page.

    Returns:
        list of tuples: A list of (article_url, speaker_name) tuples.
    """
    articles = []
    # Find all 'li' elements with class 'o-listicle__item'
    listicle_items = listing_soup.find_all('li', class_='o-listicle__item')
    for item in listicle_items:
        # Extract article URL from 'm-statement__quote' div
        quote_div = item.find('div', class_='m-statement__quote')
        if quote_div:
            link_tag = quote_div.find('a', href=True)
            if link_tag:
                href = link_tag['href']
                # Ensure the URL is absolute
                if href.startswith('/'):
                    article_url = 'https://www.politifact.com' + href
                else:
                    article_url = href
            else:
                logging.error("No <a> tag found within 'm-statement__quote' div.")
                continue
        else:
            logging.error("No 'm-statement__quote' div found in the list item.")
            continue

        # Extract speaker name from 'm-statement__name' class
        speaker_tag = item.find('a', class_='m-statement__name')
        if speaker_tag:
            speaker_name = speaker_tag.get_text(strip=True)
        else:
            logging.error(f"No speaker name found for article URL: {article_url}")
            speaker_name = 'Unknown'

        articles.append((article_url, speaker_name))
    return articles


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


def extract_main_claim(article_soup):
    """
    Extract the main claim from the article.

    Parameters:
        article_soup (BeautifulSoup): Parsed HTML of the article page.

    Returns:
        str or None: The main claim text, or None if not found.
    """
    claim_div = article_soup.find("div", class_="m-statement__quote")
    if claim_div:
        claim_text = claim_div.get_text(strip=True)
        # Remove any enclosing quotes
        claim_text = claim_text.strip('“”"')
        return claim_text
    else:
        logging.error("Main claim section 'm-statement__quote' not found in the article.")
        return None


def extract_truth_o_meter(article_soup):
    """
    Extract the Truth-O-Meter rating from the article.

    Parameters:
        article_soup (BeautifulSoup): Parsed HTML of the article page.

    Returns:
        str or None: The Truth-O-Meter rating (e.g., 'False', 'True'), or None if not found.
    """
    meter_div = article_soup.find("div", class_="m-statement__meter")
    if meter_div:
        img_tag = meter_div.find("img", alt=True)
        if img_tag:
            truth_rating = img_tag['alt'].strip().capitalize()
            return truth_rating
        else:
            logging.error("No <img> tag with 'alt' attribute found in 'm-statement__meter' div.")
            return None
    else:
        logging.error("No 'm-statement__meter' div found in the article.")
        return None


def extract_editions(article_soup):
    """
    Extract the Editions from the article.

    Parameters:
        article_soup (BeautifulSoup): Parsed HTML of the article page.

    Returns:
        list: A list of edition strings, e.g., ["PolitiFact en Español"]
    """
    editions = []
    script_tags = article_soup.find_all("script")
    for script in script_tags:
        if script.string and "Editions" in script.string:
            # Use regex to find the Editions array
            match = re.search(r"'Editions'\s*:\s*\[([^\]]+)\]", script.string)
            if match:
                editions_raw = match.group(1)
                # Extract individual editions, removing quotes and whitespace
                editions_list = re.findall(r'["\'](.*?)["\']', editions_raw)
                # Decode any Unicode escape sequences
                editions = [bytes(ed, "utf-8").decode("unicode_escape") for ed in editions_list]
                break
    if not editions:
        logging.error("Editions not found in the article.")
    return editions


def extract_justification(article_soup):
    """
    Extract the justification from the 'short-on-time' section of the article.

    Parameters:
        article_soup (BeautifulSoup): Parsed HTML of the article page.

    Returns:
        str or None: The justification text, or None if not found.
    """
    justification_div = article_soup.find("div", class_="short-on-time")
    if justification_div:
        # Attempt to extract text from list items
        list_items = justification_div.find_all("li")
        if list_items:
            # Join all list item texts into a single string
            justification_text = " ".join([li.get_text(strip=True) for li in list_items])
            return justification_text
        else:
            # If no list items, extract the direct text
            justification_text = justification_div.get_text(strip=True)
            return justification_text
    else:
        # Log if 'short-on-time' section is not found
        logging.error("No 'short-on-time' section found in the article.")
        return None


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


async def process_article(session, url, speaker_name, processed_articles):
    """
    Process a single article URL to extract unique sources, main claim, Truth-O-Meter, and justification.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session to use.
        url (str): The article URL.
        speaker_name (str): The speaker name extracted from the listing page.
        processed_articles (set): Set of already processed article URLs.

    Returns:
        dict or None: A dictionary with article details, or None if failed or non-English.
    """
    if url in processed_articles:
        return None

    article_html = await fetch(session, url)
    if not article_html:
        return None

    article_soup = BeautifulSoup(article_html, "html.parser")

    # Extract Editions to determine language
    editions = extract_editions(article_soup)
    if not editions:
        # If Editions not found, assume non-English and skip
        return None
    # Check if any non-English edition is present
    if any(edition in NON_ENGLISH_EDITIONS for edition in editions):
        # Skip non-English editions
        return None

    # Optionally, verify the language attribute in the <html> tag
    html_tag = article_soup.find("html")
    if html_tag and html_tag.has_attr("lang"):
        lang = html_tag["lang"].lower()
        if not lang.startswith("en"):
            # Skip if the language is not English
            return None
    else:
        logging.warning(f"No language attribute found in HTML tag for URL: {url}")

    # Extract unique sources
    sources = extract_unique_sources(article_soup)

    # Extract main claim
    main_claim = extract_main_claim(article_soup)

    # Extract Truth-O-Meter rating
    truth_o_meter = extract_truth_o_meter(article_soup)

    # Extract justification
    justification = extract_justification(article_soup)

    if sources or main_claim or truth_o_meter or justification:
        processed_articles.add(url)
        return {
            "Speaker": speaker_name,  # Added Speaker field
            "Article URL": url,
            "Main Claim": main_claim if main_claim else "N/A",
            "Truth-O-Meter": truth_o_meter if truth_o_meter else "N/A",
            "Justification": justification if justification else "N/A",
            "Sources": sorted(sources) if sources else []
        }
    else:
        return None


async def process_listing_page(
    session, page_number, semaphore, processed_articles, category
):
    """
    Process a single listing page to extract and process article URLs.

    Parameters:
        session (aiohttp.ClientSession): The HTTP session to use.
        page_number (int): The current listing page number.
        semaphore (asyncio.Semaphore): Semaphore to control concurrency.
        processed_articles (set): Set of already processed article URLs.
        category (str): The category being scraped.

    Returns:
        list: A list of dictionaries containing article details.
        bool: Indicator whether any articles were found on this page.
    """
    listing_url = get_listing_url(BASE_LISTING_URL, page_number, category)
    listing_html = await fetch(session, listing_url)
    if not listing_html:
        return [], False

    listing_soup = BeautifulSoup(listing_html, "html.parser")
    articles_info = extract_article_urls(listing_soup)
    if not articles_info:
        return [], False

    articles = []

    async def semaphore_process(url, speaker_name):
        async with semaphore:
            return await process_article(session, url, speaker_name, processed_articles)

    tasks = [semaphore_process(url, speaker_name) for url, speaker_name in articles_info]
    for f in tqdm_asyncio.as_completed(
        tasks,
        total=len(tasks),
        desc=f"Processing Articles on Page {page_number}",
    ):
        result = await f
        if result:
            articles.append(result)

    return articles, True


async def main(category):
    """
    The main function to orchestrate scraping for a given category.

    Parameters:
        category (str): The category to scrape.
    """
    processed_articles = set()
    all_articles_sources = []
    page_number = 1
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # Format the JSON filename based on the category
    category_filename = category.replace(" ", "_").lower()
    json_file = f"{category_filename}_unique_sources.json"

    connector = aiohttp.TCPConnector(limit_per_host=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector
    ) as session:
        while True:
            print(f"\nProcessing listing page {page_number} for category '{category}'...")
            articles, has_more = await process_listing_page(
                session, page_number, semaphore, processed_articles, category
            )
            if not has_more:
                print(f"No articles found on page {page_number}. Stopping.")
                break
            if articles:
                all_articles_sources.extend(articles)
                print(
                    f"  Extracted {len(articles)} articles from page {page_number}."
                )
            else:
                print(
                    f"  No relevant data extracted from articles on page {page_number}."
                )

            page_number += 1

    # Save the collected data to a JSON file asynchronously
    try:
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(
                json.dumps(all_articles_sources, indent=4, ensure_ascii=False)
            )
        print(f"\nData has been saved to {json_file}")
    except IOError as e:
        logging.error(f"Error writing to JSON file {json_file}: {e}")
        print(f"Error writing to JSON file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape unique sources, main claims, Truth-O-Meter ratings, and justifications from PolitiFact fact-check articles for a given category."
    )
    parser.add_argument(
        "category",
        type=str,
        help="The category to scrape (e.g., 'health', 'economy', 'politics').",
    )
    args = parser.parse_args()

    # Validate the category input if necessary
    # For example, ensure it matches expected patterns or exists on the website

    try:
        asyncio.run(main(args.category))
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        sys.exit(0)
