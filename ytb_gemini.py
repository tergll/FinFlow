import asyncio
import aiohttp
from typing import List

async def fetch_youtube_data(session: aiohttp.ClientSession, url: str) -> str:
    """
    Asynchronously fetches the HTML content of a YouTube URL.

    Args:
        session: aiohttp ClientSession for making requests.
        url: The YouTube URL to fetch.

    Returns:
        The HTML content of the page as a string, or None if an error occurred.
    """
    try:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return await response.text()
    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching {url}: {e}")
        return None


async def ingest_youtube_urls(urls: List[str]):
    """
    Asynchronously ingests data from a list of YouTube URLs.

    Args:
        urls: A list of YouTube URLs to ingest.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_youtube_data(session, url) for url in urls]
        results = await asyncio.gather(*tasks)  # Run all tasks concurrently

    # Process the results
    for url, html in zip(urls, results):
        if html:
            print(f"Successfully ingested data from {url}")
            # Here you would add your logic to process the HTML content
            # For example, you could parse it with BeautifulSoup to extract
            # specific information like title, description, etc.
        else:
            print(f"Failed to ingest data from {url}")


# Example usage:
async def main():
    youtube_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Example URL 1
        "https://www.youtube.com/watch?v=jfKfPfyJRdk",  # Example URL 2
        "https://www.youtube.com/watch?v=QH2-TGUlwu4",  # Example URL 3
        "https://www.youtube.com/watch?v=unGc-vj-LgQ",  # Example URL 4
        "https://www.youtube.com/watch?v=M-n73bbwQFM",  # Example URL 5
    ]
    await ingest_youtube_urls(youtube_urls)

if __name__ == "__main__":
    asyncio.run(main())
