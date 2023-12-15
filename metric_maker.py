import pandas as pd
import numpy as np
import networkx as nx
import sys
import gc
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    dates = [("05", "2023"), ("06", "2023"), ("07", "2023")]
    for month, year in dates:
        for i in range(1, 32):
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

def weekly_metrics():
    fecha_inicio = datetime(2023, 5, 1)
    fecha_fin = datetime(2023, 7, 31)
    dias_en_intervalo = (fecha_fin - fecha_inicio).days + 1

    conjuntos_fechas = []
    for i in range(0, dias_en_intervalo, 7):
        conjunto_actual = []
        for j in range(7):
            fecha_actual = fecha_inicio + timedelta(days=i + j)
            if fecha_actual <= fecha_fin:
                conjunto_actual.append(tuple(map(str, fecha_actual.strftime("%Y %m %d").split())))
        conjuntos_fechas.append(np.array(conjunto_actual))

    for idx in range(0, len(conjuntos_fechas) +1):
        dataframes = [
            pd.read_csv(
                f"{file[0]}-{file[1]}-{file[2]}.csv", 
                usecols=["from_address", "to_address", "value", "nonce", "gas"]
            ) for file in conjuntos_fechas[idx]
        ]
        edges = pd.concat(dataframes, ignore_index=True)

        G = nx.from_pandas_edgelist(
            edges, source="from_address", target="to_address", create_using=nx.DiGraph()
        )

        name = f"metrics_week_{idx}"
        # ruta_completa = os.path.join(os.getcwd(), "METRICAS_"+name)
        try:
            os.mkdir("WEEKLY")
        except FileExistsError:
            print("Reemplazando el archivo ya existente")
        except Exception:
            raise "Error no controlado"

        metrics = {}

        metrics, total_nodes = calculate_total_nodes(G, metrics)
        metrics, _total_edges = calculate_total_edges(G, metrics)

        # Agregaciones sobre los g- y g+
        # degree_values_plot(in_degree_values, out_degree_values, total_nodes, name)

        in_degree_values = [G.in_degree(n) for n in G.nodes]
        out_degree_values = [G.out_degree(n) for n in G.nodes]
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

        print(f"Diccionario escrito en WEEKLY/{name}.bin")
        with open(f"WEEKLY/{name}.bin", 'wb') as file:
            pickle.dump(metrics, file)

daily_metrics()
weekly_metrics()