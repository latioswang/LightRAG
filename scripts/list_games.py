import requests
from lxml import html
import glog
from pathlib import Path

def fetch_game_links():
    base_url = "https://opencritic.com/browse/ps5?page={}"
    output_file = Path("data/games_ps5_all.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_games = 0
    with open(output_file, 'w') as f:
        for page in range(1, 1001):
            try:
                response = requests.get(base_url.format(page))
                tree = html.fromstring(response.content)
                
                # Get game links
                links = tree.xpath('/html/body/app-root/div/app-browse-platform/div/div[4]/div/div[*]/div[4]/a/@href')
                if not links:
                    glog.info(f"No more games found at page {page}. Stopping.")
                    break
                
                for link in links:
                    f.write(f"{link}\n")
                
                total_games += len(links)
                glog.info(f"Page {page}: Found {len(links)} games. Total: {total_games}")
                
            except Exception as e:
                glog.error(f"Error on page {page}: {str(e)}")
                raise

if __name__ == "__main__":
    fetch_game_links()
