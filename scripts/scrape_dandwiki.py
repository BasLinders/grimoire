from curl_cffi import requests
import mwparserfromhell
import re
import os
import time

def get_dandwiki_page(page_title):
    """
    Fetches the raw wikitext content of a page from dandwiki.com using TLS impersonation.
    """
    url = "https://www.dandwiki.com/api.php"
    
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
        # Define headers that match the 'chrome' impersonation
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Referer": "https://www.dandwiki.com/",
            "Accept": "application/json, text/javascript, */*; q=0.01"
        }
        
        # Add a delay to be polite to the server
        time.sleep(2)
        
        # Use impersonate="chrome126" to target a modern browser fingerprint
        response = requests.get(url, params=params, headers=headers, impersonate="chrome110")
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
            
    except Exception as e:
        print(f"An error occurred while connecting to the API: {e}")
        return None

def clean_and_split_wikitext(raw_wikitext):
    """
    Parses raw wikitext and splits it into semantic chunks based on headers.
    """
    parsed = mwparserfromhell.parse(raw_wikitext)
    sections = parsed.get_sections(include_lead=True, flat=True)
    
    document_chunks = {}
    
    for section in sections:
        headings = section.filter_headings()
        if headings:
            title = headings[0].title.strip_code().strip()
            section.remove(headings[0]) 
        else:
            title = "Lead_Introduction"
            
        plain_text = section.strip_code()
        plain_text = re.sub(r'<[^>]+>', '', plain_text)
        plain_text = re.sub(r'\n{3,}', '\n\n', plain_text).strip()
        
        if plain_text: 
            document_chunks[title] = plain_text
            
    return document_chunks

if __name__ == "__main__":
    target_page = "5e_Classes"
    print(f"Fetching data for: {target_page}...\n")
    
    raw_content = get_dandwiki_page(target_page)
    
    if raw_content:
        print("Cleaning and chunking wikitext...")
        structured_data = clean_and_split_wikitext(raw_content)
        
        output_dir = f"dandwiki_{target_page}_dataset"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving files to ./{output_dir}/ ...")
        
        for section_title, text_content in structured_data.items():
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', section_title)
            filename = f"{safe_title}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text_content)
                
        print("All files saved successfully!")
