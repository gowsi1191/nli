import os
import json
from openai import OpenAI

# Set your OpenAI key

file_names = [
    "gemini.json",
]

def is_explicit_not(query_text):
    prompt = f"""
You are a medical NLP classifier. Only answer YES or NO.
Question: Is the following query a textbook example of an `explicit_NOT` query, with 100% confidence?

Query: "{query_text}"

Respond with YES or NO only. Do not explain.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer == "YES"
    except Exception as e:
        print("OpenAI API error:", e)
        return False

for file_path in file_names:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    retained = []
    deleted = []

    for item in data:

        query_text = item.get("query_text", "")
        print(f"→ Validating query {query_text}")
        if is_explicit_not(query_text):
            retained.append(item)
        else:
            deleted.append(item)

    # Save retained data back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(retained, f, indent=2)

    # Save deleted queries
    deleted_path = file_path.replace(".json", "_deleted.json")
    with open(deleted_path, "w", encoding="utf-8") as f:
        json.dump(deleted, f, indent=2)

    print(f"\nProcessed {file_path}:")
    print(f"  ✅ Retained (explicit_NOT): {len(retained)}")
    print(f"  ❌ Moved to deleted: {len(deleted)}")
