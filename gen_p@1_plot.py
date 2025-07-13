import os
import json
import plotly.graph_objs as go

def plot_scores_with_toggle(file_path, model_name):
    with open(file_path) as f:
        data = json.load(f)

    scores_0 = []
    scores_1 = []

    for docs in data.values():
        for doc in docs:
            if doc["relevance"] == 0:
                scores_0.append(doc["score"])
            elif doc["relevance"] == 1:
                scores_1.append(doc["score"])

    trace_0 = go.Scatter(y=scores_0, mode='markers', name='Relevance 0', marker=dict(color='red'))
    trace_1 = go.Scatter(y=scores_1, mode='markers', name='Relevance 1', marker=dict(color='green'))

    fig = go.Figure(data=[trace_0, trace_1])
    fig.update_layout(
        title=f"Interactive Score Scatter for {model_name}",
        xaxis_title="Index",
        yaxis_title="Score",
        showlegend=True
    )

    output_path = f"{model_name}_scatter.html"
    fig.write_html(output_path)
    print(f"✅ Plot saved to: {output_path}")

def main():
    folder = "/Users/L020774/Movies/heu/NLP"

    for filename in os.listdir(folder):
        if filename.startswith("roberta_scores_by_example_") and filename.endswith(".json"):
            full_path = os.path.join(folder, filename)
            model_name = filename.replace("roberta_scores_by_example_", "").replace(".json", "")
            print(f"📊 Plotting for model: {model_name}")
            plot_scores_with_toggle(full_path, model_name)

if __name__ == "__main__":
    main()
