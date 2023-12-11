import os
import pickle

directorio_principal = os.getcwd()
df_final = pd.DataFrame()


for nombre_directorio in os.listdir(directorio_principal):
    ruta_directorio = os.path.join(directorio_principal, nombre_directorio)
    if os.path.isdir(ruta_directorio) and nombre_directorio.startswith("METRICS"):
        for nombre_archivo in os.listdir(ruta_directorio):
            if nombre_archivo.lower().startswith("metrics") and nombre_archivo.lower().endswith(".pkl"):
                ruta_archivo = os.path.join(ruta_directorio, nombre_archivo)
                with open(ruta_archivo, 'rb') as file:
                    datos = pickle.load(file)
                    print(f"Datos en '{ruta_archivo}': {datos}")
                df_temporal = pd.DataFrame.from_dict(datos, orient='index').T
                df_final = pd.concat([df_final, df_temporal], ignore_index=True)
                
                