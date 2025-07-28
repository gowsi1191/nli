import time
import json
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Function to clean and truncate text to less than 250 tokens
def clean_and_truncate(text, max_tokens=400):
    if not text:
        return ""
    text = ' '.join(text.split())
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return ' '.join(tokens)

# Set up Selenium WebDriver in headless mode
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# Define all URLs and their relevance labels
all_entries =[
  {
    "relevance": 0,
    "seq": 2,
    "url": "https://www.clinicaltrials.gov/study/NCT00937066"
  },
  {
    "relevance": 0,
    "seq": 4,
    "url": "https://www.clinicaltrials.gov/study/NCT00546234"
  },
  {
    "relevance": 0,
    "seq": 3,
    "url": "https://www.clinicaltrials.gov/study/NCT05431920"
  },
  {
    "relevance": 0,
    "seq": 6,
    "url": "https://www.clinicaltrials.gov/study/NCT06029595"
  },
  {
    "relevance": 0,
    "seq": 8,
    "url": "https://www.clinicaltrials.gov/study/NCT02194699"
  },
  {
    "relevance": 1,
    "seq": 1,
    "url": "https://www.clinicaltrials.gov/study/NCT00509197"
  },
  {
    "relevance": 1,
    "seq": 5,
    "url": "https://www.clinicaltrials.gov/study/NCT00641914"
  },
  {
    "relevance": 1,
    "seq": 7,
    "url": "https://www.clinicaltrials.gov/study/NCT04865575"
  },
  {
    "relevance": 1,
    "seq": 9,
    "url": "https://www.clinicaltrials.gov/study/NCT06753214"
  },
  {
    "relevance": 1,
    "seq": 10,
    "url": "https://www.clinicaltrials.gov/study/NCT00613587"
  }
]




results = []

# Scrape each URL
for entry in all_entries:
    seq = entry["seq"]
    url = entry["url"]
    relevance = entry["relevance"]
    driver.get(url)

    # Random wait to avoid rate-limiting
    time.sleep(random.uniform(1, 2))

    def safe_get_text(by, value):
        try:
            return driver.find_element(by, value).text.strip()
        except:
            return ""

    # Extract and truncate content from specified IDs
    official_title = clean_and_truncate(safe_get_text(By.ID, "official-title-content"))
    brief_summary = clean_and_truncate(safe_get_text(By.ID, "brief-summary"))
    detailed_description = clean_and_truncate(safe_get_text(By.ID, "detailed-description"))

    full_text = " ".join([official_title, brief_summary, detailed_description])

    results.append({
        "doc_id": f"DOC{seq:03}",
        "text": full_text,
        "relevance": relevance,
        "seq": seq,
        "url": url,
        "description": ""
    })

driver.quit()

# Save as JSON
with open("clinical_trials_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Extracted and saved to clinical_trials_output.json")
