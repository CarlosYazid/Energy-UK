import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def init():
    global model
    global scaler
    global window_size
    global model_config
    
    model_config = {"protected_namespaces": ("settings_",)}

    # Leer rutas desde variables de entorno
    model_path = os.getenv("MODEL_PATH", "tensorflow_series_UK_energy_v2/")  # Ruta por defecto
    scaler_path = os.getenv("SCALER_PATH", "tensorflow_series_UK_energy_v2/scaler.pkl")  # Ruta por defecto

    try:
        # Cargar el modelo
        model = load_model(model_path)
        print(f"Modelo cargado exitosamente desde: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {e}")

    try:
        # Cargar el scaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler cargado exitosamente desde: {scaler_path}")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el scaler: {e}")

    # Configurar parámetros
    try:
        target_index = int(os.getenv("TARGET_INDEX", 0))  # Índice de la variable objetivo
        window_size = int(os.getenv("WINDOW_SIZE", 48))  # Tamaño de ventana (48 horas)
        print(f"Parámetros configurados: target_index={target_index}, window_size={window_size}")
    except ValueError as e:
        raise RuntimeError(f"Error al configurar parámetros: {e}")


# Crear ventanas de tiempo (secuencias)
def create_sequences(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)


def run(raw_data):
    try:
        logging.info(f"Datos recibidos: {raw_data}")

        # Convertir los datos de entrada en un array numpy
        input_data = np.array(json.loads(raw_data))

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Crear secuencias basadas en el tamaño de ventana
        sequences = create_sequences(scaled_data, window_size)

        # Realizar predicciones
        predictions = model.predict(sequences)

        logging.info(f"Predicción realizada: {predictions}")
        return json.dumps(predictions.tolist())
    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}", exc_info=True)
        return json.dumps({"error": "Error en la predicción.", "details": str(e)})

