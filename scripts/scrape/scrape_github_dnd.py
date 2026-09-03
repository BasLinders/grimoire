"""
Harvester for D&D GitHub repositories.
Scrapes READMEs and filters for high-value D&D content.
"""

import requests
import time
from pathlib import Path
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# OUTPUT_DIR is defined at the module level for consistency
OUTPUT_DIR = Path("data/corpus/saga/")
HEADERS = {"Accept": "application/vnd.github.v3+json"}

def get_raw_content(url: str):
    """Fetches raw content from GitHub."""
    try:
        response = requests.get(url, headers={"Accept": "application/vnd.github.v3.raw"})
        if response.status_code == 200:
            return response.text
    except Exception:
        return None
    return None

def scrape_repo_files(owner, repo):
    """Scrapes README, evaluates content, and extracts .md/.json files."""
    base_url = f"https://github.com/{owner}/{repo}/tree/main"
    
    # 1. Fetch README first
    readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
    readme_content = get_raw_content(readme_url)
    
    if not readme_content:
        # Fallback for standard README file
        readme_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README"
        readme_content = get_raw_content(readme_url)

    # 2. Pre-filter Logic
    if readme_content:
        keywords = ["stat block", "monster", "spell", "rule", "campaign", "homebrew", "dnd"]
        relevance_score = sum(1 for kw in keywords if kw.lower() in readme_content.lower())
        
        if relevance_score < 2:
            print(f"  Skipping {repo}: Low relevance (score: {relevance_score})")
            return

        # Ensure directory exists before writing
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"{repo}_README.md").write_text(readme_content, encoding="utf-8")
    else:
        print(f"  Skipping {repo}: No README found.")
        return

    # 3. Crawl remaining files
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    files = soup.select('a.Link--primary')
    for f in files:
        file_path = f.get('href')
        # Filter for relevant file types
        if file_path and (file_path.endswith('.md') or file_path.endswith('.json')):
            if 'README' in file_path:
                continue
                
            filename = file_path.split('/')[-1]
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path.split('/')[-1]}"
            content = get_raw_content(raw_url)
            if content:
                dest = OUTPUT_DIR / f"{repo}_{filename}"
                dest.write_text(content, encoding="utf-8")
                print(f"  Captured: {filename}")

def run_harvest():
    """Main execution loop for GitHub API search."""
    # Search query for D&D repositories
    url = "https://api.github.com/search/repositories?q=topic:dnd-5e&sort=stars"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"API Error: {response.status_code} - {response.text}")
        return
        
    repos = response.json().get("items", [])
    
    for r in repos:
        owner = r['owner']['login']
        name = r['name']
        print(f"Harvesting {owner}/{name}...")
        scrape_repo_files(owner, name)
        time.sleep(2) # Polite delay for API rate limits

if __name__ == "__main__":
    run_harvest()
