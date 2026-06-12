from curl_cffi import requests
import mwparserfromhell
import re
import os

def get_dandwiki_page(page_title):
    """
    Fetches the raw wikitext content of a page from dandwiki.com using its API.
    """
    url = "https://www.dandwiki.com/api.php"
    
    # It is good practice to include a descriptive User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
    
    # Parameters for the MediaWiki API
    params = {
        "action": "query",
        "format": "json",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main"
    }
    
    try:
        response = requests.get(url, params=params, impersonate="chrome")
        response.raise_for_status() 
        
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        
        for page_id, page_info in pages.items():
            if page_id == "-1":
                print(f"Error: The page '{page_title}' was not found.")
                return None
            
            # Extract the raw wikitext content
            content = page_info["revisions"][0]["slots"]["main"]["*"]
            return content
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while connecting to the API: {e}")
        return None

def clean_and_split_wikitext(raw_wikitext):
    """
    Parses raw wikitext and splits it into semantic chunks based on headers.
    Returns a dictionary of { "Header Name": "Cleaned Plain Text" }.
    """
    parsed = mwparserfromhell.parse(raw_wikitext)
    
    # get_sections() breaks the AST into distinct blocks based on header levels
    sections = parsed.get_sections(include_lead=True, flat=True)
    
    document_chunks = {}
    
    for section in sections:
        headings = section.filter_headings()
        if headings:
            # Extract the header name to use as our chunk title
            title = headings[0].title.strip_code().strip()
            # Remove the header from the body text to avoid duplication
            section.remove(headings[0]) 
        else:
            title = "Lead_Introduction"
            
        # Clean the remaining text in this section
        plain_text = section.strip_code()
        plain_text = re.sub(r'<[^>]+>', '', plain_text)
        
        # Normalize whitespace
        plain_text = re.sub(r'\n{3,}', '\n\n', plain_text).strip()
        
        # Only save chunks that actually contain data
        if plain_text: 
            document_chunks[title] = plain_text
            
    return document_chunks

if __name__ == "__main__":
    # Example: Let's fetch the main 5e Classes page
    target_page = "5e_Classes"
    print(f"Fetching data for: {target_page}...\n")
    
    raw_content = get_dandwiki_page(target_page)
    
    if raw_content:
        print("Cleaning and chunking wikitext for LLM training pipeline...\n")
        structured_data = clean_and_split_wikitext(raw_content)
        
        print(f"Success! Extracted {len(structured_data)} distinct semantic chunks.")
        print("-" * 50)
        
        # Create a directory to store the files
        output_dir = f"dandwiki_{target_page}_dataset"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving files to ./{output_dir}/ ...")
        
        for section_title, text_content in structured_data.items():
            # Sanitize the title to make it a valid filename
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', section_title)
            filename = f"{safe_title}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text_content)
                
        print("All files saved successfully!")
