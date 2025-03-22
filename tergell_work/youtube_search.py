from apify_client import ApifyClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

load_dotenv()

APIFY_API_KEY = os.getenv('APIFY_API_KEY')

def search_youtube_videos(query, max_results=50):
    """Search YouTube videos from the past 2 days using Apify"""
    try:
        client = ApifyClient(APIFY_API_KEY)
        
        # Calculate date 2 days ago
        two_days_ago = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Prepare the Actor input
        run_input = {
            "searchQueries": [query],
            "maxResults": max_results,
            "maxResultsShorts": 0,
            "maxResultStreams": 0,
            "postsFromDate": two_days_ago,
            # Removed transcript-related options
        }
        
        print("Starting YouTube scrape...")
        # Run the Actor and wait for it to finish
        run = client.actor("streamers/youtube-scraper").call(run_input=run_input)
        
        # Fetch results
        videos = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            if 'title' not in item or 'url' not in item:
                continue
                
            video = {
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'channel': item.get('channelName', ''),
                'description': item.get('description', ''),
                'publishedAt': item.get('uploadDate', ''),
                'views': item.get('viewCount', 0),
                'duration': item.get('duration', ''),
                'likes': item.get('likeCount', 0)
            }
            videos.append(video)
            
        return videos
    
    except Exception as e:
        print(f"An error occurred during scraping: {e}")
        return []

def filter_by_relevance(videos, query, top_n=5):
    """Filter videos based on title relevance using TF-IDF and cosine similarity"""
    if not videos:
        return []
        
    # Prepare texts for comparison
    titles = [video['title'] for video in videos]
    texts = titles + [query]
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate similarity between query and each title
    query_vector = tfidf_matrix[-1]
    title_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(title_vectors, query_vector)
    
    # Get top N most relevant videos
    top_indices = np.argsort(similarities.flatten())[-top_n:][::-1]
    return [videos[i] for i in top_indices]

def main():
    # Get user input
    query = input("Enter your search topic: ")
    final_results = int(input("Enter number of most relevant videos to keep (default: 5): ") or 5)
    
    # Search for videos
    print("\nSearching for recent videos...")
    videos = search_youtube_videos(query)
    
    if videos:
        print(f"\nFound {len(videos)} videos from the past 2 days.")
        print("\nFiltering for most relevant videos based on your query...")
        
        # Filter for most relevant videos
        relevant_videos = filter_by_relevance(videos, query, final_results)
        
        # Display results
        print("\nTop Relevant Videos:\n")
        for i, video in enumerate(relevant_videos, 1):
            print(f"\n{'-' * 80}")
            print(f"\nVIDEO {i}:")
            print(f"Title: {video['title']}")
            print(f"Channel: {video['channel']}")
            print(f"Published: {video['publishedAt']}")
            print(f"URL: {video['url']}")
            print(f"Views: {video['views']:,}")
            print(f"Duration: {video['duration']}")
            print(f"Likes: {video['likes']:,}")
            print(f"Description: {video['description'][:200]}...")
            print(f"\n{'-' * 80}")
    else:
        print("No videos found or an error occurred.")

if __name__ == "__main__":
    main() 