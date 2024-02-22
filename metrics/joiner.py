import os
import pickle
import pandas as pd
import re

from logger.logger import MyLogger

logger = MyLogger(__name__)

directorio_principal = os.getcwd()
df_final = pd.DataFrame()

meses = ['JULIO 2022', 'JUNIO 2022', 'MAYO 2022','JULIO 2023', 'JUNIO 2023', 'MAYO 2023']
patron_fecha = re.compile(r'\d{4}-\d{2}-\d{2}')

for mes in meses:
    for nombre_directorio in os.listdir(directorio_principal):
        ruta_directorio = os.path.join(directorio_principal, nombre_directorio)
        if os.path.isdir(ruta_directorio) and nombre_directorio.startswith(f"{mes}"):
            for nombre_directorio_act in os.listdir(ruta_directorio):
                ruta_directorio_act = os.path.join(ruta_directorio, nombre_directorio_act)
                if os.path.isdir(ruta_directorio_act) and nombre_directorio_act.startswith("METRICAS"):
                    for nombre_archivo in os.listdir(ruta_directorio_act):
                        if nombre_archivo.lower().startswith("metrics") and nombre_archivo.lower().endswith(".bin"):
                            fecha_encontrada = patron_fecha.search(nombre_archivo)
                            fecha_extraida = fecha_encontrada.group()
                            ruta_archivo = os.path.join(ruta_directorio_act, nombre_archivo)
                            with open(ruta_archivo, 'rb') as file:
                                datos = pickle.load(file)
                                logger.info(f"Datos en '{ruta_archivo}'")
                            datos['date'] = fecha_extraida
                            df_temporal = pd.DataFrame.from_dict(datos, orient='index').T
                            df_final = pd.concat([df_final, df_temporal], ignore_index=True)

df_final.to_csv("registro_metricas_ventanas.csv", index=False)
logger.info("Guardado en registro_metricas_ventanas.csv")