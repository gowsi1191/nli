from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# Function to clean and truncate text to less than 250 tokens
def clean_and_truncate(text, max_tokens=250):
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

# URLs grouped by relevance 0 with sequence numbers
relevance_groups = [
    [{"seq": 1, "url": "https://www.clinicaltrials.gov/study/NCT05359367"},
     {"seq": 3, "url": "https://www.clinicaltrials.gov/study/NCT05442463"}],
    # [{"seq": 4, "url": "https://www.clinicaltrials.gov/study/NCT04411303"},
    #  {"seq": 5, "url": "https://www.clinicaltrials.gov/study/NCT05846958"}]
]

results = []

# Scrape each URL
for group in relevance_groups:
    for entry in group:
        seq = entry["seq"]
        url = entry["url"]
        driver.get(url)
        time.sleep(5)  # Let the page load

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
            "doc_id": "DOC907",
            "text": full_text,
            "relevance": 0,
            "seq": seq,
            "url": url,
            "description": ""
        })

driver.quit()

# Output result as JSON
print(json.dumps(results, indent=2))
