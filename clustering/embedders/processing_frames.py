import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, BatchNormalization, Lambda
from clustering.embedders.all_v1.Generator import GeneratorTriplet
from clustering.embedders.all_v2.LossV2 import TripletCustom
import os
from datetime import datetime, timedelta


def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
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
    return ids

def create_aux_model(ids):
    # el 1 representa a cualquier indice, un tensor de una dimensión
    input_aux = Input(1)

    # el 128 es arbitrario, podría ser de cualquier valor
    x = Embedding(len(ids), 128)(input_aux)

    # TODO: investigar que es esto
    x = Dense(64, activation='tanh')(x)

    # lleva todos los valores entre -1 y 1
    output_aux = BatchNormalization()(x)

    model_aux = Model(input_aux, output_aux)
    model_aux.summary()
    return model_aux

def create_model(model_aux):
    # son los 3 inputs que representan las 3 cabezas de la red, lo que se conoce como siamesa
    input_layer_anchor = Input(1)
    input_layer_positive = Input(1)
    input_layer_negative = Input(1)

    x_a = model_aux(input_layer_anchor)
    x_p = model_aux(input_layer_positive)
    x_n = model_aux(input_layer_negative)

    # un tensor es ... investigar bien. Puede ser cualquier cosa, literalmente.
    merged_output = Lambda(lambda tensors: tf.stack(tensors, axis=-1))([x_a, x_p, x_n])

    model = Model([input_layer_anchor, input_layer_positive, input_layer_negative], merged_output)
    model.summary()
    return model

def create_generator(df, ids):
    return GeneratorTriplet(df, ids, 64)

def compile_model(model):
    loss = TripletCustom()    

    model.compile(
        optimizer=Adam(2e-3),
        loss=loss
    )

def train_model(model, generator):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(generator, epochs=1000, callbacks=[callback])

def pipeline(files):
    df = read_files(files)
    cleaned_df = clean_nodes(df)
    addresses_ids = create_ids(cleaned_df)
    model_aux = create_aux_model(addresses_ids)
    model = create_model(model_aux)
    generator = create_generator(cleaned_df, addresses_ids)
    compile_model(model)
    train_model(model, generator)

def pipeline_v2(df):
    cleaned_df = clean_nodes(df)
    addresses_ids = create_ids(cleaned_df)
    model_aux = create_aux_model(addresses_ids)
    model = create_model(model_aux)
    generator = create_generator(cleaned_df, addresses_ids)
    compile_model(model)
    train_model(model, generator)
    return model_aux, addresses_ids


# Definir los nombres de los meses y el número de días en cada mes
dias_por_mes = {"June": 30, "July": 31}

# Función para obtener los archivos de un directorio dado
def obtener_archivos(ruta_directorio):
    return [os.path.join(ruta_directorio, archivo) for archivo in os.listdir(ruta_directorio)]

# Función para obtener las fechas en una ventana temporal
def obtener_fechas_ventana(fecha_inicial, ventana):
    fechas = []
    for i in range(0, dias_por_mes[fecha_inicial.strftime("%B")], ventana):
        ventana_inicio = fecha_inicial + timedelta(days=i)
        ventana_fin = min(ventana_inicio + timedelta(days=ventana - 1), datetime(fecha_inicial.year, fecha_inicial.month, dias_por_mes[fecha_inicial.strftime("%B")]))
        fechas.append((ventana_inicio.strftime("%Y-%m-%d"), ventana_fin.strftime("%Y-%m-%d")))
    return fechas

def generate_dates(src, months):
    # Construir el array deseado
    array_datos = []

    for year in range(2023, 2024):  # ajustar los años según sea necesario
        for mes in months:
            ruta_mes = f"{src}/{year}/{mes}"
            archivos_mes = obtener_archivos(ruta_mes)
            array_mes = [[archivo] for archivo in archivos_mes]
            array_datos.extend(array_mes)

            # Agregar la ventana del mes completo
            archivos_mes_completo = [os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv") for day in range(1, dias_por_mes[mes] + 1)]
            array_datos.append(archivos_mes_completo)

            # Agregar las ventanas de 7 días
            for i in range(0, dias_por_mes[mes], 7):
                ventana_inicio = datetime(year, months.index(mes) + 1, i + 1)
                ventana_fin = min(ventana_inicio + timedelta(days=6), datetime(year, months.index(mes) + 1, dias_por_mes[mes]))

                # Asegurarnos de que los días estén dentro del mes actual
                archivos_ventana_7_dias = []
                for day in range(ventana_inicio.day, ventana_fin.day + 1):
                    if day <= dias_por_mes[mes]:
                        archivos_ventana_7_dias.append(os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv"))

                array_datos.append(archivos_ventana_7_dias)

            # Agregar las ventanas de 15 días
            for i in range(0, dias_por_mes[mes], 15):
                ventana_inicio = datetime(year, months.index(mes) + 1, i + 1)
                ventana_fin = min(ventana_inicio + timedelta(days=14), datetime(year, months.index(mes) + 1, dias_por_mes[mes]))

                # Asegurarnos de que los días estén dentro del mes actual y no se dupliquen
                archivos_ventana_15_dias = []
                for day in range(ventana_inicio.day, ventana_fin.day + 1):
                    if day <= dias_por_mes[mes] and day > i:
                        archivos_ventana_15_dias.append(os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv"))
                
                if len(archivos_ventana_15_dias) == 1:
                    continue

                array_datos.append(archivos_ventana_15_dias)

# pipeline
def main():
    set_gpu()
    months = ["July"]
    src = "../../../datos"
    generated_files = generate_dates(src, months)
    for files in generated_files:
        pipeline(files)


# # Definir los nombres de los meses y el número de días en cada mes
# meses = ["july"]
# dias_por_mes = {"june": 30, "july": 31}
# indice_por_mes = {"june": 6, "july": 7}

# # Función para obtener los archivos de un directorio dado
# def obtener_archivos(ruta_directorio):
#     return [os.path.join(ruta_directorio, archivo) for archivo in os.listdir(ruta_directorio)]

# # Función para obtener las fechas en una ventana temporal
# def obtener_fechas_ventana(fecha_inicial, ventana):
#     fechas = []
#     for i in range(ventana):
#         fecha = fecha_inicial + timedelta(days=i)
#         fechas.append(fecha.strftime("%Y-%m-%d"))
#     return fechas

# # Construir el array deseado
# array_datos = []

# for year in range(2023, 2024):
#     for mes in meses:
#         ruta_mes = f"../../../datos/{year}/{mes}"
#         archivos_mes = obtener_archivos(ruta_mes)
#         array_mes = [[archivo] for archivo in archivos_mes]
#         array_datos.extend(array_mes)

#         for dia in range(1, dias_por_mes[mes] + 1):
#             fecha_actual = datetime(year, indice_por_mes[mes], dia)
            
#             # Ventana de una semana
#             ventana_una_semana = obtener_fechas_ventana(fecha_actual, 7)
#             archivos_semana = [os.path.join(ruta_mes, f"{fecha}.csv") for fecha in ventana_una_semana]
#             array_datos.append(archivos_semana)

#             # Ventana de quince días
#             ventana_quince_dias = obtener_fechas_ventana(fecha_actual, 15)
#             archivos_quince_dias = [os.path.join(ruta_mes, f"{fecha}.csv") for fecha in ventana_quince_dias]
#             array_datos.append(archivos_quince_dias)

# print(sorted(array_datos))
# print(len(array_datos))
        


# import os
# from datetime import datetime, timedelta

# # Definir los nombres de los meses y el número de días en cada mes
# meses = ["July"]
# dias_por_mes = {"June": 30, "July": 31}

# # Función para obtener los archivos de un directorio dado
# def obtener_archivos(ruta_directorio):
#     return [os.path.join(ruta_directorio, archivo) for archivo in os.listdir(ruta_directorio)]

# # Función para obtener las fechas en una ventana temporal
# def obtener_fechas_ventana(fecha_inicial, ventana):
#     fechas = []
#     for i in range(0, dias_por_mes[fecha_inicial.strftime("%B")], ventana):
#         ventana_inicio = fecha_inicial + timedelta(days=i)
#         ventana_fin = min(ventana_inicio + timedelta(days=ventana - 1), datetime(fecha_inicial.year, fecha_inicial.month, dias_por_mes[fecha_inicial.strftime("%B")]))
#         fechas.append((ventana_inicio.strftime("%Y-%m-%d"), ventana_fin.strftime("%Y-%m-%d")))
#     return fechas

# # Construir el array deseado
# array_datos = []

# for year in range(2023, 2024):  # ajustar los años según sea necesario
#     for mes in meses:
#         ruta_mes = f"../../../datos/{year}/{mes}"
#         archivos_mes = obtener_archivos(ruta_mes)
#         array_mes = [[archivo] for archivo in archivos_mes]
#         array_datos.extend(array_mes)

#         # Agregar la ventana del mes completo
#         archivos_mes_completo = [os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv") for day in range(1, dias_por_mes[mes] + 1)]
#         array_datos.append(archivos_mes_completo)

#         # Agregar las ventanas de 7 días
#         for i in range(0, dias_por_mes[mes], 7):
#             ventana_inicio = datetime(year, meses.index(mes) + 1, i + 1)
#             ventana_fin = min(ventana_inicio + timedelta(days=6), datetime(year, meses.index(mes) + 1, dias_por_mes[mes]))

#             # Asegurarnos de que los días estén dentro del mes actual
#             archivos_ventana_7_dias = []
#             for day in range(ventana_inicio.day, ventana_fin.day + 1):
#                 if day <= dias_por_mes[mes]:
#                     archivos_ventana_7_dias.append(os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv"))

#             array_datos.append(archivos_ventana_7_dias)

#         # Agregar las ventanas de 15 días
#         for i in range(0, dias_por_mes[mes], 15):
#             ventana_inicio = datetime(year, meses.index(mes) + 1, i + 1)
#             ventana_fin = min(ventana_inicio + timedelta(days=14), datetime(year, meses.index(mes) + 1, dias_por_mes[mes]))

#             # Asegurarnos de que los días estén dentro del mes actual y no se dupliquen
#             archivos_ventana_15_dias = []
#             for day in range(ventana_inicio.day, ventana_fin.day + 1):
#                 if day <= dias_por_mes[mes] and day > i:
#                     archivos_ventana_15_dias.append(os.path.join(ruta_mes, f"{year}-{mes}-{day}.csv"))
            
#             if len(archivos_ventana_15_dias) == 1:
#                 continue

#             array_datos.append(archivos_ventana_15_dias)