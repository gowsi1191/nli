import os
import json

# === Paths ===
original_dir = "/Users/L020774/Documents/nlp_nli/nli/json/data"
new_dir = os.path.join(original_dir, "new")

total_docs = 0
correct = 0
mismatch = 0
skipped = 0

for i in range(1, 54):
    orig_file = os.path.join(original_dir, f"query{i}.json")
    new_file = os.path.join(new_dir, f"query{i}.json")

    if not os.path.exists(orig_file) or not os.path.exists(new_file):
        print(f"⚠️ Skipping query{i}: File missing")
        skipped += 1
        continue

    with open(orig_file) as f1, open(new_file) as f2:
        try:
            orig_data = json.load(f1)
            new_data = json.load(f2)
        except Exception as e:
            print(f"⚠️ Skipping query{i}: Error parsing JSON — {e}")
            skipped += 1
            continue

    # Handle list wrapping if needed
    if isinstance(orig_data, list):
        orig_data = orig_data[0]
    if isinstance(new_data, list):
        new_data = new_data[0]

    orig_docs = orig_data.get("documents", [])
    new_docs = new_data.get("documents", [])

    if len(orig_docs) != len(new_docs):
        print(f"⚠️ Mismatch in number of documents in query{i}")
        skipped += 1
        continue

    for j, (orig_doc, new_doc) in enumerate(zip(orig_docs, new_docs), start=1):
        total_docs += 1
        if orig_doc["doc_id"] != new_doc["doc_id"]:
            print(f"⚠️ Doc ID mismatch in query{i} doc{j}")
            skipped += 1
            continue

        if orig_doc["relevance"] == new_doc["relevance"]:
            correct += 1
        else:
            mismatch += 1

# === Summary ===
print("\n📊 Relevance Comparison Summary")
print(f"✅ Correct (unchanged): {correct}")
print(f"❌ Mismatch (changed):  {mismatch}")
print(f"⚠️ Skipped:             {skipped}")
print(f"📄 Total Compared:      {total_docs}")
