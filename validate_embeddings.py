import os
import json
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime

def load_embeddings(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    vectors = []
    chunk_indexes = []

    for entry in data.get("llm_output", []):
        vec = entry.get("vector")
        if vec and isinstance(vec, list):
            vectors.append(vec)
            chunk_indexes.append(entry.get("chunk_index", len(vectors)))

    return np.array(vectors), chunk_indexes

def validate_shape_and_length(vectors):
    lengths = [len(vec) for vec in vectors]
    consistent = all(length == lengths[0] for length in lengths)
    return consistent, lengths[0] if consistent else -1

def calculate_average_similarity(vectors):
    sims = []
    for i in range(len(vectors) - 1):
        sim = cosine_similarity([vectors[i]], [vectors[i+1]])[0][0]
        sims.append(sim)
    return np.mean(sims), sims

def plot_pca(vectors, output_path):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], cmap='viridis', alpha=0.7)
    plt.title('PCA Projection of Embeddings')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def log_results(log_path, summary):
    with open(log_path, 'w') as log_file:
        for key, value in summary.items():
            log_file.write(f"{key}: {value}\n")

def main(json_path, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'embedding_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    pca_plot = os.path.join(log_dir, f'pca_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

    vectors, indexes = load_embeddings(json_path)
    summary = {}

    if vectors.size == 0:
        summary["Status"] = "No vectors found."
        log_results(log_file, summary)
        return

    consistent, vec_length = validate_shape_and_length(vectors)
    avg_sim, similarities = calculate_average_similarity(vectors)
    plot_pca(vectors, pca_plot)

    summary.update({
        "Status": "Success",
        "Total Chunks": len(vectors),
        "Vector Length": vec_length,
        "All Vectors Same Length": consistent,
        "Average Similarity Between Adjacent Chunks": round(avg_sim, 4),
        "PCA Plot Saved": pca_plot
    })

    log_results(log_file, summary)
    print(f"Validation complete. Summary logged to {log_file}")
    print(f"PCA plot saved to {pca_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and visualize embedding output JSON.")
    parser.add_argument("--file", required=True, help="Path to JSON embedding file.")
    parser.add_argument("--logdir", default="logs", help="Directory to save logs and plots.")
    args = parser.parse_args()

    main(args.file, args.logdir)
