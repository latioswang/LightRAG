"""
This script is used to answer questions about the the input is a reddit post about the feelings of the games the user want. The final output is a list of games with reasons why they are good.

The workflow of this script is:
1. Read the reddit post
2. Convert reddit post into 3 google search queries that can best generate a list of potential games
3. Use all of the google search results to generate a list of potential games
3. Convert the reddit post into a scoring system that can score the games, possible scoring data sources are:
    - Steam reviews
    - Metacritic reviews
4. For each game, apply the scoring system to get a score
4. Output the games and reasons why they are good
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from typing import List, Dict
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameRecommender:
    def __init__(self, api_key: str):
        """Initialize the GameRecommender with necessary API keys."""
        openai.api_key = api_key
        self.steam_api_url = "https://store.steampowered.com/api"
        self.metacritic_base_url = "https://www.metacritic.com/game"

    def generate_search_queries(self, reddit_post: str) -> List[str]:
        """Generate search queries based on the Reddit post content."""
        prompt = f"""
        Based on this Reddit post, generate 3 different search queries to find video game recommendations:
        {reddit_post}
        
        Return only the 3 queries, one per line.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        queries = response.choices[0].message.content.strip().split('\n')
        return queries[:3]

    def search_games(self, queries: List[str]) -> List[str]:
        """Search for games using Google search."""
        games = set()
        for query in queries:
            search_results = search(query + " game review", num_results=5)
            for result in search_results:
                # Extract game names from search results
                # This is a simplified version; you might want to add more sophisticated parsing
                game_name = self._extract_game_name(result)
                if game_name:
                    games.add(game_name)
        return list(games)

    def _extract_game_name(self, url: str) -> str:
        """Extract game name from URL - placeholder for more sophisticated parsing."""
        # This is a simplified version
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Add your game name extraction logic here
            return ""
        except:
            return ""

    def create_scoring_system(self, reddit_post: str) -> Dict:
        """Create a scoring system based on the Reddit post."""
        prompt = f"""
        Create a scoring system based on this Reddit post:
        {reddit_post}
        
        Return a JSON object with weights for different aspects:
        - gameplay (0-1)
        - story (0-1)
        - graphics (0-1)
        - multiplayer (0-1)
        - difficulty (0-1)
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.choices[0].message.content)

    def score_game(self, game: str, scoring_system: Dict) -> Dict:
        """Score a game based on the scoring system."""
        steam_data = self._get_steam_data(game)
        metacritic_data = self._get_metacritic_data(game)
        
        score = 0
        reasons = []
        
        # Calculate score based on available data and scoring system
        # This is a placeholder for more sophisticated scoring logic
        if steam_data:
            score += steam_data.get('rating', 0) * 0.5
            reasons.append(f"Steam rating: {steam_data.get('rating', 0)}")
            
        if metacritic_data:
            score += metacritic_data.get('score', 0) * 0.5
            reasons.append(f"Metacritic score: {metacritic_data.get('score', 0)}")
            
        return {
            'game': game,
            'score': score,
            'reasons': reasons
        }

    def _get_steam_data(self, game: str) -> Dict:
        """Get game data from Steam API."""
        # Implement Steam API integration
        return {}

    def _get_metacritic_data(self, game: str) -> Dict:
        """Get game data from Metacritic."""
        # Implement Metacritic scraping
        return {}

def main():
    parser = argparse.ArgumentParser(description='Game recommendation system based on Reddit posts')
    parser.add_argument('--reddit_post', type=str, help='Path to file containing Reddit post', default='reddit.txt')
    parser.add_argument('--api-key', type=str, help='OpenAI API key', default="")
    arg  = parser.parse_args()

    try:
        # Read Reddit post
        with open(args.reddit_post, 'r') as f:
            reddit_post = f.read()

        # Initialize recommender
        recommender = GameRecommender(args.api_key)

        # Generate search queries
        queries = recommender.generate_search_queries(reddit_post)
        logger.info(f"Generated queries: {queries}")

        # Search for games
        games = recommender.search_games(queries)
        logger.info(f"Found games: {games}")

        # Create scoring system
        scoring_system = recommender.create_scoring_system(reddit_post)
        logger.info(f"Created scoring system: {scoring_system}")

        # Score games
        results = []
        for game in games:
            score_data = recommender.score_game(game, scoring_system)
            results.append(score_data)

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Output results
        print("\nRecommended Games:")
        for result in results[:5]:  # Top 5 games
            print(f"\n{result['game']} (Score: {result['score']:.2f})")
            print("Reasons:")
            for reason in result['reasons']:
                print(f"- {reason}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

