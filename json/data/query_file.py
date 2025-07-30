import os
import json

# Base directory to save files
output_dir = "json/data"
os.makedirs(output_dir, exist_ok=True)

# Sample queries
queries = [
    "Asthma control strategies excluding corticosteroids",
    "Management of rheumatoid arthritis excluding TNF-alpha inhibitors",
    "Treatment of chronic pain not involving NSAIDs or opioids",
    "Lowering intraocular pressure without prostaglandin analogs",
    "Preventing cardiovascular events without aspirin therapy",
    "Eradicating Helicobacter pylori without clarithromycin-based regimens",
    "Managing ADHD without stimulant medications",
    "Treating insomnia excluding benzodiazepines or Z-drugs",
    "Reducing triglyceride levels without omega-3 fatty acids or fibrates",
    "Handling acute gout flares without colchicine or NSAIDs",
    "Preventing migraine recurrence without CGRP inhibitors",
    "Addressing hormone-sensitive breast cancer without aromatase inhibitors"
]

# Create JSON files
for i, query in enumerate(queries, start=25):
    content = [{
        "id": "12",
        "query": query,
        "documents": []
    }]
    file_path = os.path.join(output_dir, f"query{i}.json")
    with open(file_path, "w") as f:
        json.dump(content, f, indent=2)

print("✅ 12 query files created.")
