# TRABAJO-PROFESIONAL-7599-

# ejecuciones:

### Ejecución 1: `poetry run python3 main.py --level INFO`

Crea un grafo fake para crear embeddings de dos formas diferentes. 

1. El primer caso es con una red neuronal (...desarrollar...)
2. El segundo caso es haciendo uso de Node2Vec

Luego de haber construido los embeddings lo que hacemos es aplicar kmeans y hdbscan sobre estos para comparar la performance de nuestro modelo con respecto a Node2Vec, visualizando además con UMAP. Finalmente con un ID de ejecución logueamos los resultados finales y con qué parámetro hicimos cada paso.

# Módulos

## Metrics_maker

Este script se utiliza para extraer metricas de un grafo definido por un csv.

    - Se carga el csv y se infla el grafo.
    - Se saca cada metrica y se la va guardando de un diccionario 
    - Al final se guarda en disco el diccionario en una carpeta METRICAS_<nombre csv>/metrics_<nombre csv>.bin

## Joiner 

Este script se encarga de agarrar los binarios hechos por el metrics_maker y crear un csv nuevo que contenga las metricas para cada ventana calculada en un registro.


## Clustering

Consideraciones:
    - Deberiamos incluir count por pares de transacciones y monto total a lo que da el generador para poder usarlo inteligentemente en la loss.
    - Deberiamos usar un threshold para excluir pares de address. Idea inicial, hacer histograma de count por pares y excluir los que esten por debajo del p15.
