import argparse
import asyncio
import json
import logging
import re
import sys
import urllib.parse

import aiofiles
import aiohttp
import tldextract
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

# Configure logging for error handling
logging.basicConfig(
    filename="scraper_errors.log",
    level=logging.ERROR,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Constants (SPEAKER will be set dynamically)
BASE_LISTING_URL = (
    "https://www.politifact.com/factchecks/list/?page={page}&speaker={speaker}"
)
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
]

# Mapping of domains to standard source names
SOURCE_DOMAIN_MAP = {
    "nytimes.com": "The New York Times",
    "cnn.com": "Cable News Network",
    "washingtonpost.com": "The Washington Post",
    "apnews.com": "The Associated Press",
    "associatedpress.com": "The Associated Press",
    "whitehouse.gov": "White House",
    "politico.com": "Politico",
    "c-span.org": "C-SPAN",
    "nbcnews.com": "NBC News",
    "foxnews.com": "Fox News",
    "npr.org": "National Public Radio",
    "crsreports.congress.gov": "Congressional Research Service",
}


def get_listing_url(base_url, page_number, speaker):
    """Generate a single listing URL based on the page number and speaker."""
    return base_url.format(page=page_number, speaker=speaker)


def extract_article_urls(listing_soup, speaker):
    """Extract article URLs from a listing page's BeautifulSoup object."""
    article_urls = set()
    article_url_pattern = re.compile(
        rf"^https://www\.politifact\.com/factchecks/\d{{4}}/[a-z]{{3}}/\d{{2}}/{re.escape(speaker)}/[^/]+/?$",
        re.IGNORECASE,
    )
    for a_tag in listing_soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/"):
            href = "https://www.politifact.com" + href
        elif not href.startswith("http"):
            href = "https://www.politifact.com/" + href.lstrip("/")
        if article_url_pattern.match(href):
            article_urls.add(href)
    return article_urls


def extract_unique_sources(article_soup):
    """
    Extract unique sources and their corresponding URLs from the 'Our Sources' section.

    Returns:
        dict: A dictionary with standard source names as keys and URLs or "N/A" as values
    """
    # Initialize the sources dictionary with standard source names and "N/A" as default values
    sources = {
        source_name: "N/A" for source_name in SOURCE_DOMAIN_MAP.values()
    }

    sources_section = article_soup.find("section", id="sources")
    if sources_section:
        source_paragraphs = sources_section.find_all("p")
        for p in source_paragraphs:
            links = p.find_all("a")
            if links:
                for link in links:
                    source_url = link.get("href", "")
                    if source_url:
                        # Ensure URL is absolute
                        if source_url.startswith("/"):
                            source_url = (
                                "https://www.politifact.com" + source_url
                            )
                        elif not source_url.startswith(
                            ("http://", "https://")
                        ):
                            source_url = "https://" + source_url.lstrip("/")

                        # Parse the domain using tldextract
                        ext = tldextract.extract(source_url)
                        domain = f"{ext.domain}.{ext.suffix}".lower()

                        # Check if the domain matches any in our mapping
                        if domain in SOURCE_DOMAIN_MAP:
                            standard_name = SOURCE_DOMAIN_MAP[domain]
                            sources[standard_name] = source_url
            else:
                # No links, skip
                continue
    else:
        logging.error("No 'Our Sources' section found in the article.")
    return sources


def extract_main_claim(article_soup):
    """Extract the main claim from the article."""
    claim_div = article_soup.find("div", class_="m-statement__quote")
    if claim_div:
        return claim_div.get_text(strip=True)
    else:
        logging.error(
            "Main claim section 'm-statement__quote' not found in the article."
        )
        return None


def extract_truth_o_meter(article_soup):
    """Extract the Truth-O-Meter rating from the article."""
    script_tags = article_soup.find_all("script")
    for script in script_tags:
        if script.string and "Truth-O-Meter" in script.string:
            match = re.search(
                r"'Truth-O-Meter'\s*:\s*'([^']+)'", script.string
            )
            if match:
                return match.group(1)
    logging.error("Truth-O-Meter rating not found in the article.")
    return None


def extract_editions(article_soup):
    """Extract the Editions from the article."""
    editions = []
    script_tags = article_soup.find_all("script")
    for script in script_tags:
        if script.string and "Editions" in script.string:
            match = re.search(r"'Editions'\s*:\s*\[([^\]]+)\]", script.string)
            if match:
                editions_raw = match.group(1)
                editions_list = re.findall(r'["\'](.*?)["\']', editions_raw)
                editions = [
                    bytes(ed, "utf-8").decode("unicode_escape")
                    for ed in editions_list
                ]
                break
    if not editions:
        logging.error("Editions not found in the article.")
    return editions


def extract_justification(article_soup):
    """Extract the justification from the 'short-on-time' section."""
    justification_div = article_soup.find("div", class_="short-on-time")
    if justification_div:
        list_items = justification_div.find_all("li")
        if list_items:
            justification_text = " ".join(
                [li.get_text(strip=True) for li in list_items]
            )
            return justification_text
        else:
            justification_text = justification_div.get_text(strip=True)
            return justification_text
    else:
        logging.error("No 'short-on-time' section found in the article.")
        return None


async def fetch(session, url):
    """Asynchronously fetch the content of a URL."""
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()
            # Attempt to get the encoding from the response
            encoding = response.charset or "utf-8"
            try:
                return await response.text(encoding=encoding)
            except UnicodeDecodeError as e:
                logging.error(f"UnicodeDecodeError for URL {url}: {e}")
                # Fallback: decode with 'utf-8' and replace undecodable bytes
                return await response.text(encoding="utf-8", errors="replace")
    except aiohttp.ClientError as e:
        logging.error(f"Error fetching URL {url}: {e}")
    except asyncio.TimeoutError:
        logging.error(f"Timeout fetching URL {url}")
    return None


async def fetch_title_and_text(session, url):
    """Fetch the page and extract the title and text."""
    html = await fetch(session, url)
    if not html:
        return None, None
    soup = BeautifulSoup(html, "html.parser")
    title = (
        soup.title.string.strip()
        if soup.title and soup.title.string
        else "N/A"
    )
    # For text, extract all paragraphs
    paragraphs = soup.find_all("p")
    text = (
        " ".join([p.get_text(strip=True) for p in paragraphs])
        if paragraphs
        else "N/A"
    )
    return title, text


async def process_article(session, url, processed_articles, speaker):
    """Process a single article URL to extract details."""
    if url in processed_articles:
        return None

    article_html = await fetch(session, url)
    if not article_html:
        return None

    article_soup = BeautifulSoup(article_html, "html.parser")

    editions = extract_editions(article_soup)
    if not editions or any(
        edition in NON_ENGLISH_EDITIONS for edition in editions
    ):
        return None

    html_tag = article_soup.find("html")
    if html_tag and html_tag.has_attr("lang"):
        lang = html_tag["lang"].lower()
        if not lang.startswith("en"):
            return None
    else:
        logging.warning(
            f"No language attribute found in HTML tag for URL: {url}"
        )

    sources = extract_unique_sources(article_soup)
    main_claim = extract_main_claim(article_soup)
    truth_o_meter = extract_truth_o_meter(article_soup)
    justification = extract_justification(article_soup)

    if sources or main_claim or truth_o_meter or justification:
        processed_articles.add(url)

        # Fetch title and text for each source URL
        source_titles_texts = {}
        for source_name, source_url in sources.items():
            if source_url != "N/A":
                title, text = await fetch_title_and_text(session, source_url)
                # If title or text couldn't be fetched, set to "N/A"
                source_titles_texts[source_name] = {
                    "URL": source_url,
                    "Title": title if title else "N/A",
                    "Text": text if text else "N/A",
                }
            else:
                source_titles_texts[source_name] = {
                    "URL": "N/A",
                    "Title": "N/A",
                    "Text": "N/A",
                }

        return {
            "Speaker": speaker,
            "Article URL": url,
            "Main Claim": main_claim if main_claim else "N/A",
            "Truth-O-Meter": truth_o_meter if truth_o_meter else "N/A",
            "Justification": justification if justification else "N/A",
            "Sources": sources,  # Now contains dict with standard source names and URLs or "N/A"
            "Source Titles and Texts": source_titles_texts,
        }
    else:
        return None


async def process_listing_page(
    session, page_number, semaphore, processed_articles, speaker
):
    """Process a single listing page to extract and process article URLs."""
    listing_url = get_listing_url(BASE_LISTING_URL, page_number, speaker)
    listing_html = await fetch(session, listing_url)
    if not listing_html:
        return [], False

    listing_soup = BeautifulSoup(listing_html, "html.parser")
    article_urls = extract_article_urls(listing_soup, speaker)

    if not article_urls:
        return [], False

    articles = []

    async def semaphore_process(url):
        async with semaphore:
            return await process_article(
                session, url, processed_articles, speaker
            )

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


async def main(speaker):
    """Main function to orchestrate scraping for a given speaker."""
    processed_articles = set()
    all_articles_sources = []
    page_number = 1
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    speaker_filename = speaker.replace(" ", "_").lower()
    json_file = f"{speaker_filename}_sources_with_urls.json"

    connector = aiohttp.TCPConnector(limit_per_host=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(
        headers=HEADERS, connector=connector
    ) as session:
        while True:
            print(
                f"\nProcessing listing page {page_number} for speaker '{speaker}'..."
            )
            articles, has_more = await process_listing_page(
                session, page_number, semaphore, processed_articles, speaker
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
        description="Scrape sources with URLs and other details from PolitiFact fact-check articles for a given speaker."
    )
    parser.add_argument(
        "speaker",
        type=str,
        help="The speaker category to scrape (e.g., 'instagram-posts', 'twitter', 'facebook').",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.speaker))
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
