import pandas as pd
import numpy as np
import networkx as nx
import sys
import gc
import pickle
import os
import matplotlib.pyplot as plt


if len(sys.argv) < 2:
    print("Agregar nombre del archivo csv de transacciones a leer")
    sys.exit(1)

nombre_archivo = sys.argv[1]

nombre = os.path.splitext(nombre_archivo)[0]

ruta_completa = os.path.join(os.getcwd(), "METRICAS_"+nombre)
os.mkdir(ruta_completa)

edges = pd.read_csv(
    nombre_archivo, usecols=["from_address", "to_address", "value", "nonce", "gas"]
)
G = nx.from_pandas_edgelist(
    edges, source="from_address", target="to_address", create_using=nx.DiGraph()
)

metrics = {}

# Cantiadad de nodos totales
total_nodes = G.number_of_nodes()
metrics["total_nodes"] = total_nodes

# Cantidad de aristas totales
total_edges = G.number_of_edges()
metrics["total_edges"] = total_edges

# Agregaciones sobre los g- y g+
in_degree_values = [G.in_degree(n) for n in G.nodes]
out_degree_values = [G.out_degree(n) for n in G.nodes]
in_degree_values_plot = sorted(in_degree_values, reverse=True)
out_degree_values_plot = sorted(out_degree_values, reverse=True)
in_degree_values_plot = np.array(in_degree_values_plot)
out_degree_values_plot = np.array(out_degree_values_plot)
in_degree_values_plot = in_degree_values_plot / total_nodes
out_degree_values_plot = out_degree_values_plot / total_nodes
in_degree_values_plot = np.cumsum(in_degree_values_plot)
out_degree_values_plot = np.cumsum(out_degree_values_plot)
plt.plot(in_degree_values_plot, label="in-degree")
plt.plot(out_degree_values_plot, label="out-degree")
plt.legend()
plt.title('Distribucion acumulativa de grados de entrada y salida')
plt.savefig(f'METRICAS_{nombre}/distribucion_acum.png')

metrics["mean_in"] = np.mean(in_degree_values)
metrics["max_in"] = np.max(in_degree_values)
metrics["min_in"] = np.min(in_degree_values)

metrics["mean_out"] = np.mean(out_degree_values)
metrics["max_out"] = np.max(out_degree_values)
metrics["min_out"] = np.min(out_degree_values)

out_degree_values = ""
in_degree_values = ""
gc.collect()


# PageRank
print("Calculando PageRank")
pagerank = nx.pagerank(G)
pagerank_for_metrics = list(pagerank.values())

metrics["mean_pagerank"] = np.mean(pagerank_for_metrics)
metrics["min_pagerank"] = np.min(pagerank_for_metrics)
metrics["max_pagerank"] = np.max(pagerank_for_metrics)

pagerank = ""
pagerank_for_metrics = ""
gc.collect()

# HITS
print("Calculando HITS")
hits = nx.hits(G)
hub_scores, authority_scores = hits

hub_scores_list = [x for x in hub_scores.values()]
authority_scores_list = [x for x in authority_scores.values()]

metrics["mean_hub_score"] = np.mean(hub_scores_list)
metrics["min_hub_score"] = np.min(hub_scores_list)
metrics["max_hub_score"] = np.max(hub_scores_list)

metrics["mean_authority_scores"] = np.mean(authority_scores_list)
metrics["min_authority_scores"] = np.min(authority_scores_list)
metrics["max_authority_scores"] = np.max(authority_scores_list)

hub_scores_list = ""
authority_scores_list = ""
hits = ""
gc.collect()

# # Katz
# print("Calculando Katz")
# katz = nx.katz_centrality(G, tol=1e-02)
# katz_score = list(katz.values())

# metrics["mean_katz_score"] = np.mean(katz_score)
# metrics["min_katz_score"] = np.min(katz_score)
# metrics["max_katz_score"] = np.max(katz_score)

# katz_score = ""
# katz = ""
# gc.collect()


# Metricas sobre columnas

print("Calculando metricas sobre columnas")

metrics["mean_value"] = edges["value"].mean()
metrics["max_value"] = edges["value"].max()
print("Calculando max_value")
metrics["mean_nonce"] = edges["nonce"].mean()
metrics["max_nonce"] = edges["nonce"].max()
metrics["min_nonce"] = edges["nonce"].min()
print("Calculando max_nonce")
metrics["mean_gas"] = edges["gas"].mean()
metrics["max_gas"] = edges["gas"].max()
metrics["min_gas"] = edges["gas"].min()
print("Calculando max_gas")


nombre = os.path.splitext(nombre_archivo)[0]
print(f"Diccionario escrito en METRICAS_{nombre}/metrics_{nombre}.bin")
with open(f"METRICAS_{nombre}/metrics_{nombre}.bin", 'wb') as file:
    pickle.dump(metrics, file)
