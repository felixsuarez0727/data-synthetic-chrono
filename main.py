"""
Punto de entrada principal para el proyecto de generación de datos sintéticos
con Chronos-T5 para series temporales
"""

import os
import argparse
import logging
import yaml
import pandas as pd
import torch
import numpy as np
from datetime import datetime

from src.data_preprocessing import DataPreprocessor
from src.synthetic_generator import SyntheticDataGenerator
from src.evaluation import TimeSeriesEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(config):
    """Configura los directorios necesarios"""
    os.makedirs(config['data']['raw_path'], exist_ok=True)
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    os.makedirs(config['data']['synthetic_path'], exist_ok=True)

def load_config(config_path="config/config.yaml"):
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def generate_synthetic_data(input_file, output_file=None, config_path="config/config.yaml"):
    """
    Genera datos sintéticos a partir de un archivo de entrada
    
    Args:
        input_file: Nombre del archivo de entrada en data/raw/
        output_file: Nombre del archivo de salida (si es None, se genera automáticamente)
        config_path: Ruta al archivo de configuración
    """
    start_time = datetime.now()
    logger.info(f"Iniciando proceso de generación de datos sintéticos usando Chronos-T5 a partir de {input_file}")
    
    # Cargar configuración
    config = load_config(config_path)
    setup_directories(config)
    
    # Si no se especifica archivo de salida, generar nombre
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"synthetic_{timestamp}.csv"
    
    # 1. Preprocesamiento
    logger.info("Paso 1: Preprocesamiento de datos de series temporales")
    preprocessor = DataPreprocessor(config_path)
    
    try:
        # Cargar datos
        df = preprocessor.load_data(input_file)
        processed_df = preprocessor.preprocess_data(df)
        train_df, test_df = preprocessor.split_train_test(processed_df)
        
        # 2. Preparar series temporales para Chronos-T5
        logger.info("Paso 2: Preparando series temporales para Chronos-T5")
        time_series_data = preprocessor.prepare_for_chronos(train_df)
        
        # Verificar que tenemos datos de series temporales
        if not time_series_data or len(time_series_data) == 0:
            raise ValueError("No se pudieron extraer series temporales de los datos proporcionados")
        
        logger.info(f"Se han preparado {len(time_series_data)} series temporales")
        
        # Mostrar información sobre las series
        for i, series in enumerate(time_series_data):
            logger.info(f"  Serie {i}: {len(series)} puntos, rango: [{series.min().item():.2f}, {series.max().item():.2f}]")
        
        # 3. Generación de pronósticos sintéticos
        logger.info("Paso 3: Generando pronósticos sintéticos con Chronos-T5")
        generator = SyntheticDataGenerator(config_path)
        synthetic_df = generator.generate_synthetic_dataset(
            time_series_data,
            processed_df,
            output_file
        )
        
        # 4. Evaluación de calidad de los pronósticos
        logger.info("Paso 4: Evaluando calidad de los pronósticos")
        evaluator = TimeSeriesEvaluator(config_path)
        evaluation_results = evaluator.run_evaluation("test_data.csv", output_file)
        
        # Mostrar resultados de evaluación
        if 'average_metrics' in evaluation_results:
            metrics = evaluation_results['average_metrics']
            logger.info("Resultados de evaluación:")
            if 'avg_rmse' in metrics:
                logger.info(f"  RMSE promedio: {metrics['avg_rmse']:.4f}")
            if 'avg_mae' in metrics:
                logger.info(f"  MAE promedio: {metrics['avg_mae']:.4f}")
            if 'avg_mape' in metrics:
                logger.info(f"  MAPE promedio: {metrics['avg_mape']:.2f}%")
            if 'avg_r2' in metrics:
                logger.info(f"  R² promedio: {metrics['avg_r2']:.4f}")
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds() / 60
        logger.info(f"Proceso completado en {elapsed_time:.2f} minutos")
        logger.info(f"Datos sintéticos guardados en data/synthetic/{output_file}")
        logger.info(f"Visualizaciones y evaluaciones guardadas en data/synthetic/evaluation/")
        
        return True
        
    except Exception as e:
        logger.error(f"Error durante el proceso: {str(e)}")
        raise
        
def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Generador de datos sintéticos con Chronos-T5')
    parser.add_argument('--input', '-i', required=True, help='Archivo de datos de entrada (en data/raw/)')
    parser.add_argument('--output', '-o', help='Nombre del archivo de salida (opcional)')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Ruta al archivo de configuración')
    
    args = parser.parse_args()
    
    try:
        generate_synthetic_data(args.input, args.output, args.config)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())