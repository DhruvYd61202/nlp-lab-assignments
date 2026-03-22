import requests
from bs4 import BeautifulSoup
import re
import time

def scrape_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'[^a-z0-9\s.]', '', text)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error Scraping {url} : {e}")
        return " "

def prepare_corpus(urls, window_size=2):
    master_corpus = " "
    print("Starting corpus aggregation...")
    for url in urls:
        print(f"Fetching: {url}")
        master_corpus += scrape_text_from_url(url) + " "
        time.sleep(1)

    tokens = master_corpus.split()
    vocab = sorted(set(tokens))
    V = len(vocab)
    
    word_to_ix = {w: i for i, w in enumerate(vocab)}
    ix_to_word = {i: w for w, i in word_to_ix.items()}

    cbow_data = []
    for i in range(window_size, len(tokens) - window_size):
        context = tokens[i-window_size:i] + tokens[i+1:i+window_size+1]
        target = tokens[i]
        cbow_data.append((context, target))
        
    return cbow_data, word_to_ix, ix_to_word, V