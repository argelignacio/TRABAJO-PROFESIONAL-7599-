import pandas as pd
import numpy as np
import networkx as nx
import sys
import gc
import pickle
import os
import matplotlib.pyplot as plt

def calculate_total_nodes(G, metrics):
    # Cantiadad de nodos totales
    total_nodes = G.number_of_nodes()
    metrics["total_nodes"] = total_nodes
    return metrics, total_nodes

def calculate_total_edges(G, metrics):
    # Cantidad de aristas totales
    total_edges = G.number_of_edges()
    metrics["total_edges"] = total_edges
    return metrics, total_edges

def degree_values_plot(in_degree_values, out_degree_values, total_nodes, name):
    in_degree_values_plot = sorted(in_degree_values, reverse=True)
    in_degree_values_plot = np.array(in_degree_values_plot)
    in_degree_values_plot = in_degree_values_plot / total_nodes
    in_degree_values_plot = np.cumsum(in_degree_values_plot)

    out_degree_values_plot = sorted(out_degree_values, reverse=True)
    out_degree_values_plot = np.array(out_degree_values_plot)
    out_degree_values_plot = out_degree_values_plot / total_nodes
    out_degree_values_plot = np.cumsum(out_degree_values_plot)
    plt.plot(in_degree_values_plot, label="in-degree")
    plt.plot(out_degree_values_plot, label="out-degree")
    plt.legend()
    plt.title('Distribucion acumulativa de grados de entrada y salida')
    plt.savefig(f'METRICAS_{name}/distribucion_acum.png')

def metrics_aggregation(metrics, data, alias):
    metrics[f"mean_{alias}"] = np.mean(data)
    metrics[f"std_{alias}"] = np.std(data)
    metrics[f"max_{alias}"] = np.max(data)
    metrics[f"min_{alias}"] = np.min(data)
    return metrics

def pagerank_metrics(G, metrics):
    # PageRank
    print("Calculando PageRank")
    pagerank = nx.pagerank(G)
    pagerank_for_metrics = list(pagerank.values())
    metrics = metrics_aggregation(metrics, pagerank_for_metrics, "pagerank")
    pagerank = ""
    pagerank_for_metrics = ""
    gc.collect()
    return metrics

def hits_metrics(G, metrics):
    print("Calculando HITS")
    hits = nx.hits(G)
    hub_scores, authority_scores = hits
    hub_scores_list = [x for x in hub_scores.values()]
    authority_scores_list = [x for x in authority_scores.values()]
    metrics = metrics_aggregation(metrics, hub_scores_list, "hub_score")
    metrics = metrics_aggregation(metrics, authority_scores_list, "authority_scores")
    hub_scores_list = ""
    authority_scores_list = ""
    hits = ""
    gc.collect()
    return metrics

def column_metrics(metrics, data, column_name):
    print(f"agregando columna {column_name}")
    metrics[f"mean_{column_name}"] = data[column_name].mean()
    metrics[f"std_{column_name}"] = data[column_name].std()
    metrics[f"max_{column_name}"] = data[column_name].max()
    metrics[f"min_{column_name}"] = data[column_name].min()
    return metrics

def daily_metrics():
    dates = [("05", "2022", 31), ("06", "2022", 30), ("07", "2022", 31)]
    for month, year, day_month in dates:
        in_degree_values_month = []
        out_degree_values_month = []
        for i in range(1, day_month + 1):
            if i < 10:
                name_archivo = f'{year}-{month}-0{i}.csv'
            else:
                name_archivo = f'{year}-{month}-{i}.csv'

            name = os.path.splitext(name_archivo)[0]

            ruta_completa = os.path.join(os.getcwd(), "METRICAS_"+name)
            try:
                os.mkdir(ruta_completa)
            except FileExistsError:
                print("Reemplazando el archivo ya existente")
            except Exception:
                raise "Error no controlado"

            edges = pd.read_csv(
                name_archivo, usecols=["from_address", "to_address", "value", "nonce", "gas"]
            )
            G = nx.from_pandas_edgelist(
                edges, source="from_address", target="to_address", create_using=nx.DiGraph()
            )

            metrics = {}

            metrics, total_nodes = calculate_total_nodes(G, metrics)
            metrics, _total_edges = calculate_total_edges(G, metrics)

            # Agregaciones sobre los g- y g+
            in_degree_values = [G.in_degree(n) for n in G.nodes]
            out_degree_values = [G.out_degree(n) for n in G.nodes]
            in_degree_values_month.extend(in_degree_values)
            out_degree_values_month.extend(out_degree_values)
            degree_values_plot(in_degree_values, out_degree_values, total_nodes, name)

            metrics = metrics_aggregation(metrics, in_degree_values, "in")
            metrics = metrics_aggregation(metrics, out_degree_values, "out")
            out_degree_values = ""
            in_degree_values = ""
            gc.collect()

            pagerank_metrics(G, metrics)
            hits_metrics(G, metrics)
            edges["value"] = edges["value"].astype(float)
            metrics = column_metrics(metrics, edges, "value")
            metrics = column_metrics(metrics, edges, "nonce")
            metrics = column_metrics(metrics, edges, "gas")

            name = os.path.splitext(name_archivo)[0]
            print(f"Diccionario escrito en METRICAS_{name}/metrics_{name}.bin")
            with open(f"METRICAS_{name}/metrics_{name}.bin", 'wb') as file:
                pickle.dump(metrics, file)
        

        plt.hist(in_degree_values_month, bins=25, alpha=0.7, color='skyblue', edgecolor='black', log=True)
        plt.title(f'Grados de Entrada en Escala Logarítmica- {month}-{year}')
        plt.xlabel('Grado del Nodo')
        plt.ylabel('Frecuencia')
        plt.savefig(f'histograma_entrada_{month}_{year}.png')
        plt.clf()

        # Plotear histograma de grados de salida
        plt.hist(out_degree_values_month, bins=25, alpha=0.7, color='firebrick', edgecolor='black', log=True)
        plt.title(f'Grados de Salida en Escala Logarítmica - {month}-{year}')
        plt.xlabel('Grado del Nodo')
        plt.ylabel('Frecuencia')
        plt.savefig(f'histograma_salida_{month}_{year}.png')
        plt.clf()

daily_metrics()