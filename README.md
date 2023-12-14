# TRABAJO-PROFESIONAL-7599-

## Metrics_maker

Este script se utiliza para extraer metricas de un grafo definido por un csv.

    - Se carga el csv y se infla el grafo.
    - Se saca cada metrica y se la va guardando de un diccionario 
    - Al final se guarda en disco el diccionario en una carpeta METRICAS_<nombre csv>/metrics_<nombre csv>.bin

## Joiner 

Este script se encarga de agarrar los binarios hechos por el metrics_maker y crear un csv nuevo que contenga las metricas para cada ventana calculada en un registro.
