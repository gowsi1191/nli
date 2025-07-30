import time
import json
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Token truncation function
def clean_and_truncate(text, max_tokens=400):
    if not text:
        return ""
    text = ' '.join(text.split())
    return ' '.join(text.split()[:max_tokens])

# Safe text extractor
def safe_get_text(driver, by, value):
    try:
        return driver.find_element(by, value).text.strip()
    except:
        return ""

# Headless browser setup
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Load demo.json
with open("demo.json", "r") as f:
    demo_data = json.load(f)

start_id = 35

for idx, query_entry in enumerate(demo_data):
    query_id = str(start_id + idx)
    query_text = query_entry.get("query", "")
    documents = query_entry.get("documents", [])

    scraped_docs = []

    for doc in documents:
        url = doc.get("url")
        seq = doc.get("seq")
        relevance = doc.get("relevance", 1)

        driver.get(url)
        time.sleep(random.uniform(1.5, 2.5))

        title = clean_and_truncate(safe_get_text(driver, By.ID, "official-title-content"))
        brief = clean_and_truncate(safe_get_text(driver, By.ID, "brief-summary"))
        detail = clean_and_truncate(safe_get_text(driver, By.ID, "detailed-description"))

        full_text = " ".join([title, brief, detail])

        scraped_docs.append({
            "doc_id": f"DOC{seq:03}",
            "text": full_text,
            "relevance": relevance,
            "seq": seq,
            "url": url,
            "description": ""
        })

    output = [{
        "id": query_id,
        "query": query_text,
        "documents": scraped_docs
    }]

    output_file = f"query{query_id}.json"
    with open(output_file, "w") as out_f:
        json.dump(output, out_f, indent=2)

    print(f"✅ Saved: {output_file}")

driver.quit()
print("🚀 All queries processed.")
