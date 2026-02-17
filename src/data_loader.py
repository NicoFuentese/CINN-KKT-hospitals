import pandas as pd
import numpy as np
import torch

#Carga y procesa los datos, puedes adjuntarle la fecha que se quiere analizar
def load_and_preprocess_data(csv_path, device, target_date='2023-02-01', num_samples=16, buffer_time=20.0):
    print(f"Cargando dataset desde: {csv_path}")
    df = pd.read_csv(csv_path)

    cols_needed = ['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 
                   'Ingreso Recuperación', 'Salida Recuperación', 'Tiempo Intervención', 'Tiempo Recuperación']
    df_clean = df.dropna(subset=cols_needed).copy()

    # Convertir a datetime
    for col in ['Ingreso Pabellón', 'Inicio Intervención', 'Término Intervención', 'Ingreso Recuperación', 'Salida Recuperación']:
        df_clean[col] = pd.to_datetime(df_clean[col])

    # Calcular duraciones
    df_clean['dur_pre'] = (df_clean['Inicio Intervención'] - df_clean['Ingreso Pabellón']).dt.total_seconds() / 60.0
    df_clean['dur_qx'] = df_clean['Tiempo Intervención']
    df_clean['dur_post'] = df_clean['Tiempo Recuperación']

    # Filtrar validos
    df_clean = df_clean[(df_clean['dur_pre'] > 0) & (df_clean['dur_qx'] > 0) & (df_clean['dur_post'] > 0)]

    # Seleccion de dia especifico a simular
    df_clean['date'] = df_clean['Ingreso Pabellón'].dt.date
    df_day = df_clean[df_clean['date'] == pd.to_datetime(target_date).date()].head(num_samples).reset_index(drop=True)

    if len(df_day) == 0:
        raise ValueError(f"No hay datos suficientes para la fecha {target_date}.")

    # Matrices de Tiempos
    p_time_medical_np = df_day[['dur_pre', 'dur_qx', 'dur_post']].values.astype(np.float32)
    J, I = p_time_medical_np.shape
    R = 4  # Capacidad fija de 4 pabellones por etapa

    p_time_medical = torch.tensor(p_time_medical_np, device=device)
    p_time_occupancy = p_time_medical + buffer_time

    print(f"Dataset listo: {J} pacientes procesados en {I} etapas (R={R}).")
    print(f"dia analizado con mas carga y duracion de etapas: {df_day["Ingreso Pabellón"].dt.strftime("%d-%m-%y").iloc[0]}")
    return p_time_medical, p_time_occupancy, J, I, R