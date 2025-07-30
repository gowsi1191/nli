import json

def normalize_fields(doc):
    """Ensure we always have 'entailment', 'neutral', 'contradiction', 'combined_en' fields."""
    # Handle both long-form and short-form keys
    entailment = doc.get("entailment", doc.get("e", 0.0))
    neutral = doc.get("neutral", doc.get("n", 0.0))
    contradiction = doc.get("contradiction", doc.get("c", 0.0))
    combined_en = doc.get("combined_en", entailment + neutral)
    return entailment, neutral, contradiction, combined_en

def print_sorted_docs(filepath, label):
    with open(filepath) as f:
        data = json.load(f)

    for example_id, docs in data.items():
        print(f"\n📄 {label} → {example_id} — Sorted by Entailment ↓\n")
        # Normalize and sort
        sorted_docs = sorted(docs, key=lambda x: normalize_fields(x)[0], reverse=True)

        for doc in sorted_docs:
            entailment, neutral, contradiction, combined_en = normalize_fields(doc)
            print(f"🧾 Doc ID: {doc.get('doc_id')} | Relevance: {doc.get('relevance')}")
            print(f"   Entailment:    {entailment:.6f}")
            print(f"   Neutral:       {neutral:.6f}")
            print(f"   Combined e+n:  {combined_en:.6f}")
            print(f"   Best Hypothesis: {doc.get('best_hypothesis', '')[:120]}...")
            print()

# Load and print for both files
print_sorted_docs("pushed.json", label="PUSHED")
print_sorted_docs("puactual.json", label="BASELINE")
