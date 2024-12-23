import os
import json
import glog as log
import argparse
import requests
import asyncio
from pathlib import Path
from typing import List, Optional
from lightrag import LightRAG
from lightrag.llm import gpt_4o_mini_complete
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, field_validator
from neo4j import GraphDatabase

class ImageSrc(BaseModel):
    og: str
    sm: str
    lg: str

class Outlet(BaseModel):
    id: int
    name: str
    isContributor: bool
    imageSrc: Optional[ImageSrc] = None

class ScoreFormatOption(BaseModel):
    _id: str
    val: int
    label: str

class ScoreFormat(BaseModel):
    id: int
    name: str
    shortName: str
    scoreDisplay: Optional[str]
    isNumeric: bool
    isSelect: bool
    isStars: Optional[bool] = None
    numDecimals: Optional[int] = None
    base: Optional[int] = None
    options: Optional[List[ScoreFormatOption]] = None

class Game(BaseModel):
    id: int
    name: str

class Author(BaseModel):
    id: int
    name: str
    image: Optional[bool] = None
    _id: str
    # missing if self.image is false
    imageSrc: Optional[ImageSrc] = None

class Platform(BaseModel):
    id: int
    name: str
    shortName: str
    _id: str

class GameReview(BaseModel):
    Outlet: Outlet
    ScoreFormat: ScoreFormat
    game: Game
    overrideRecommendation: bool
    _id: str
    isChosen: bool
    title: Optional[str] = None
    publishedDate: datetime
    externalUrl: str
    snippet: Optional[str] = None
    language: str
    score: Optional[float] = None
    npScore: Optional[int]
    Authors: List[Author]
    Platforms: List[Platform]
    alias: Optional[str] = None
    medianAtTimeOfReview: Optional[float] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    v: Optional[int] = Field(None, alias='__v')

    class Config:
        strict = False
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @model_validator(mode='after')
    def convert_float_scores(self):
        # Convert float scores to integers if they exist
        if isinstance(self.score, float):
            self.score = int(self.score)
        if isinstance(self.npScore, float):
            self.npScore = int(self.npScore)
        return self

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch game reviews and insert into LightRAG')
    parser.add_argument('--input-file', type=str, default='./data/games/game_urls_ps5.txt',
                       help='Input file containing game URLs')
    parser.add_argument('--cache-dir', type=str, default='./data/games/opencritic_cache',
                       help='Directory to cache game reviews')
    parser.add_argument('--working-dir', type=str, default='./data/games/lightrag',
                       help='LightRAG working directory')
    parser.add_argument('--neo4j-uri', type=str, default='neo4j://localhost:7687',
                       help='Neo4j URI')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--neo4j-password', type=str, default='password',
                       help='Neo4j password')
    parser.add_argument('--clear-db', action='store_true',
                       help='Clear the LightRAG database before insertion')
    return parser.parse_args()

def extract_game_ids(input_file: str) -> List[str]:
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    game_ids = []
    for line in lines:
        # Extract game ID from URL path like /game/9136/baldurs-gate-3
        parts = line.strip().split('/')
        if len(parts) >= 3:
            game_ids.append(parts[2])
    
    return game_ids

def safe_validate_reviews(data: List[dict]) -> List[GameReview]:
    """Safely validate review data, logging errors and only raising if too many failures."""
    validated_reviews = []
    error_count = 0
    
    for review_data in data:
        try:
            review = GameReview.model_validate(review_data)
            validated_reviews.append(review)
        except Exception as e:
            error_count += 1
            log.error(f"Failed to validate review: {str(e)[:200]}...")
    
    # Raise exception if more than 10% of reviews failed validation
    if error_count > len(data) * 0.1:
        raise ValueError(f"Too many validation errors: {error_count}/{len(data)} reviews failed")
    
    return validated_reviews

def fetch_game_reviews(game_id: str, cache_dir: str) -> List[GameReview]:
    cache_file = Path(cache_dir) / f"{game_id}.json"
    
    # Return cached data if exists
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return safe_validate_reviews(data)
    
    # Fetch from API if not cached
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
        'DNT': '1'
    }
    
    url = f'https://api.opencritic.com/api/review/game/{game_id}/all'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    # Cache the response
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(data, f)
    
    # Convert to GameReview objects with error handling
    return safe_validate_reviews(data)

def prepare_review_text(reviews: List[GameReview]) -> str:
    """Convert reviews into a structured text format for RAG"""
    text_parts = []
    
    for review in reviews:
        text_parts.append(f"Game Review: {review.game.name}\n")
        text_parts.append(f"Title: {review.title}\n")
        text_parts.append(f"Outlet: {review.Outlet.name}\n")
        text_parts.append(f"Author(s): {', '.join(author.name for author in review.Authors)}\n")
        text_parts.append(f"Published: {review.publishedDate.strftime('%Y-%m-%d')}\n")
        text_parts.append(f"Platform(s): {', '.join(platform.name for platform in review.Platforms)}\n")
        text_parts.append(f"Score: {review.score} ({review.ScoreFormat.scoreDisplay})\n")
        text_parts.append(f"Review Text: {review.snippet}\n")
        text_parts.append(f"URL: {review.externalUrl}\n")
        text_parts.append("-" * 80 + "\n")
    
    return "\n".join(text_parts)

async def gpt4o_mini_complete_with_retries(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    keyword_extraction=False, 
    *args, 
    **kwargs
):
    try: 
        return await gpt_4o_mini_complete(
            prompt, 
            system_prompt,
            history_messages,
            keyword_extraction,
            *args, 
            openai_kwargs={"max_retries": 0}, 
            **kwargs
        )
    except Exception as e:
        log.error(f"Error in gpt4o_mini_complete: prompt={...} tokens={len(prompt) // 4}, system_prompt={system_prompt}, "
                     f"history_messages={history_messages}, keyword_extraction={keyword_extraction}, "
                     f"args={args}, kwargs={kwargs}")
        raise

def clear_neo4j_database():
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Delete all nodes and relationships
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

async def main():
    args = parse_args()
    
    # Set up Neo4j environment variables
    os.environ["NEO4J_URI"] = args.neo4j_uri
    os.environ["NEO4J_USERNAME"] = args.neo4j_user
    os.environ["NEO4J_PASSWORD"] = args.neo4j_password
    
    # Create working directory
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Clear database if requested
    if args.clear_db:
        log.info("Clearing Neo4j database...")
        clear_neo4j_database()
        # Also clear the LightRAG working directory
        if os.path.exists(args.working_dir):
            import shutil
            shutil.rmtree(args.working_dir)
            os.makedirs(args.working_dir)
        log.info("Database and working directory cleared")
    
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=gpt4o_mini_complete_with_retries,
        # graph_storage="Neo4JStorage",
        graph_storage="NetworkXStorage",
        enable_llm_cache=True,
        addon_params={
            "entity_types": [
                "game", 
                "game_publisher", 
                "review", 
                "review_author", 
                "platform", 
                "outlet", 
                "score_format",
            ],
        }
    )
    
    
    # Get game IDs from input file
    game_ids = extract_game_ids(args.input_file)
    
    # Process each game
    for game_id in game_ids:
        print(f"Processing game ID: {game_id}")
        
        # Fetch reviews (cached if available)
        reviews = fetch_game_reviews(game_id, args.cache_dir)
        
        # Prepare text for RAG
        review_text = prepare_review_text(reviews)
        # Insert into LightRAG
        await rag.ainsert(review_text)
        
        print(f"Successfully processed game ID: {game_id}")
    
    print("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())
