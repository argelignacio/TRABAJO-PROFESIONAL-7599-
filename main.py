import os
import sys
# sys.path.insert(0, os.path.abspath(".."))
from vs_embedders.vs import run_all
from fake_graph.Fake_graph import GeneratorFakeGraph
from clustering.hdbscan.hdbscan_plot import run_hdbscan
from clustering.kmeans.kmeans_plot import run_kmeans

def main():
    df = GeneratorFakeGraph.generate_fake_graph_df(6, 300, 0.85, 3000)
    df.to_csv("fake_graph.csv", index=False)
    run_all(df)

    run_kmeans()
    run_hdbscan()

main()