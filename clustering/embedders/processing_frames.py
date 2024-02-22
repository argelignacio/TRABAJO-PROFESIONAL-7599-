import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
from clustering.embedders.all_v2.GeneratorV2 import GeneratorTriplet
from clustering.embedders.all_v1.Loss import EuclideanLoss
from clustering.embedders.all_v2.ModelV2 import ModelBuilder
import os
from datetime import datetime, timedelta
from logger.logger import MyLogger

logger = MyLogger(__name__)

def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"GPU set: {logical_gpus}")
        except RuntimeError as e:
            print(e)

def read_files(files):
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

def clean_nodes(df):
    df = df[~df.to_address.isna()]
    df = df[~df.from_address.isna()]
    return df[df["from_address"] != df["to_address"]]

def create_ids(df):
    ids = {}
    for i, id in enumerate(set(df['from_address']).union(set(df['to_address']))):
        ids[id] = i
    logger.info("Ids created")
    return ids

def create_generator(df, ids):
    generator = GeneratorTriplet(df, ids, 64)
    logger.info("Generator created")
    return generator

def train_model(model, generator):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(generator, epochs=1000, callbacks=[callback])

def pipeline(files):
    df = read_files(files)
    cleaned_df = clean_nodes(df)
    addresses_ids = create_ids(cleaned_df)

    model = ModelBuilder(addresses_ids, EuclideanLoss, Adam)

    generator = create_generator(cleaned_df, addresses_ids)
    embeddings = model.compile_model().fit(generator).get_embeddings()
    return embeddings

def pipeline_v2(df):
    set_gpu()
    cleaned_df = clean_nodes(df)
    addresses_ids = create_ids(cleaned_df)

    logger.info("Creating model")
    model = ModelBuilder(addresses_ids, EuclideanLoss, Adam)
    
    generator = create_generator(cleaned_df, addresses_ids)
    embeddings = model.compile_model().fit(generator).get_embeddings()
    return embeddings, addresses_ids


dias_por_mes = {"June": 30, "July": 31}

def obtener_archivos(ruta_directorio):
    return [os.path.join(ruta_directorio, archivo) for archivo in os.listdir(ruta_directorio)]

def obtener_fechas_ventana(fecha_inicial, ventana):
    fechas = []
    for i in range(0, dias_por_mes[fecha_inicial.strftime("%B")], ventana):
        ventana_inicio = fecha_inicial + timedelta(days=i)
        ventana_fin = min(ventana_inicio + timedelta(days=ventana - 1), datetime(fecha_inicial.year, fecha_inicial.month, dias_por_mes[fecha_inicial.strftime("%B")]))
        fechas.append((ventana_inicio.strftime("%Y-%m-%d"), ventana_fin.strftime("%Y-%m-%d")))
    return fechas

def generate_dates(src, months):
    array_datos = []

    for year in range(2023, 2024):
        for mes in months:
            ruta_mes = f"{src}/{year}/{mes}"
            archivos_mes = obtener_archivos(ruta_mes)
            array_mes = [[archivo] for archivo in archivos_mes]
            array_datos.extend(array_mes)

            archivos_mes_completo = [os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv") for day in range(1, dias_por_mes[mes] + 1)]
            array_datos.append(archivos_mes_completo)

            for i in range(0, dias_por_mes[mes], 7):
                ventana_inicio = datetime(year, months.index(mes) + 1, i + 1)
                ventana_fin = min(ventana_inicio + timedelta(days=6), datetime(year, months.index(mes) + 1, dias_por_mes[mes]))

                archivos_ventana_7_dias = []
                for day in range(ventana_inicio.day, ventana_fin.day + 1):
                    if day <= dias_por_mes[mes]:
                        archivos_ventana_7_dias.append(os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv"))

                array_datos.append(archivos_ventana_7_dias)

            for i in range(0, dias_por_mes[mes], 15):
                ventana_inicio = datetime(year, months.index(mes) + 1, i + 1)
                ventana_fin = min(ventana_inicio + timedelta(days=14), datetime(year, months.index(mes) + 1, dias_por_mes[mes]))

                archivos_ventana_15_dias = []
                for day in range(ventana_inicio.day, ventana_fin.day + 1):
                    if day <= dias_por_mes[mes] and day > i:
                        archivos_ventana_15_dias.append(os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv"))
                
                if len(archivos_ventana_15_dias) == 1:
                    continue

                array_datos.append(archivos_ventana_15_dias)

def main():
    set_gpu()
    months = ["July"]
    src = "../../../datos"
    generated_files = generate_dates(src, months)
    for files in generated_files:
        pipeline(files)