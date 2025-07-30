import os
import json
from openai import OpenAI
from tqdm import tqdm

# === Initialize OpenAI client ===

# === Input/output directories ===
input_dir = "/Users/L020774/Documents/nlp_nli/nli/json/data"
output_dir = os.path.join(input_dir, "new")
os.makedirs(output_dir, exist_ok=True)

# === GPT Model to Use ===
GPT_MODEL = "gpt-4"

# === Function to call OpenAI Chat Completion ===
def ask_openai(query, doc_text):
    prompt = (
        f"Query: {query}\n\n"
        f"Document: {doc_text}\n\n"
        "Determine whether this document is relevant **considering negation logic in the query**. "
        "Respond with one word only: 'Relevant' or 'Irrelevant'."
    )
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        if answer == "relevant":
            return 0
        elif answer == "irrelevant":
            return 1
        else:
            print(f"⚠️ Unexpected response: '{answer}'")
            return None
    except Exception as e:
        print(f"⚠️ OpenAI API error: {e}")
        return None

# === Main loop to process query1.json to query53.json ===
for i in range(1, 54):
    file_name = f"query{i}.json"
    file_path = os.path.join(input_dir, file_name)

    if not os.path.exists(file_path):
        print(f"❌ File missing: {file_path}")
        continue

    with open(file_path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Could not parse JSON in {file_path}")
            continue

    # Handle wrapped list structure
    if isinstance(data, list):
        data = data[0]

    query = data.get("query")
    documents = data.get("documents", [])
    if not query or not documents:
        print(f"⚠️ Skipping query{i}: missing query or documents.")
        continue

    updated_docs = []

    print(f"\n🔍 Validating Query {i}: {query}")
    for doc in tqdm(documents, desc=f"Query {i}", unit="doc"):
        doc_text = doc.get("text", "")
        if not doc_text:
            continue

        predicted_relevance = ask_openai(query, doc_text)
        if predicted_relevance is not None:
            doc["relevance"] = predicted_relevance
        updated_docs.append(doc)

    # Save updated version
    updated_data = {
        "id": str(i),
        "query": query,
        "documents": updated_docs
    }

    out_path = os.path.join(output_dir, file_name)
    with open(out_path, "w") as fout:
        json.dump(updated_data, fout, indent=2)

print("\n✅ Relevance correction completed and saved in `new/` folder.")
