"""
Módulo para generar datos sintéticos usando el modelo Chronos-T5
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import logging
import yaml
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import re

from src.model_loader import ChronosModelLoader
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Clase para generar datos sintéticos usando Chronos-T5
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Inicializa el generador con la configuración especificada
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_path = self.config['data']['processed_path']
        self.synthetic_path = self.config['data']['synthetic_path']
        
        # Crear directorios si no existen
        os.makedirs(self.synthetic_path, exist_ok=True)
        
        # Cargar el modelo Chronos-T5
        self.model_loader = ChronosModelLoader(config_path)
        self.model, self.tokenizer = self.model_loader.load_model()
        self.generation_params = self.model_loader.get_generation_params()
        
        # Configuración de generación
        self.batch_size = self.config['generation']['batch_size']
        self.num_synthetic_samples = self.config['generation']['num_synthetic_samples']
        self.seed = self.config['generation']['seed']
        
        # Establecer semilla para reproducibilidad
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
    
    def generate(self, time_series_data: Union[List[torch.Tensor], torch.Tensor]) -> List[np.ndarray]:
        """
        Genera pronósticos sintéticos a partir de datos de series temporales
        
        Args:
            time_series_data: Lista de tensores o tensor con datos de series temporales
            
        Returns:
            Lista de arrays numpy con pronósticos sintéticos generados
        """
        logger.info(f"Iniciando generación de datos sintéticos para series temporales...")
        
        # Si es una lista de tensores individuales, procesarlos uno por uno
        if isinstance(time_series_data, list):
            all_forecasts = []
            
            for i, series in enumerate(tqdm(time_series_data, desc="Generando datos sintéticos")):
                # Generar pronósticos utilizando el pipeline de Chronos
                forecasts = self.model_loader.generate_forecasts(
                    series, 
                    prediction_length=self.generation_params.get('max_length', 24)
                )
                all_forecasts.append(forecasts)
            
            logger.info(f"Generados pronósticos sintéticos para {len(all_forecasts)} series temporales")
            return all_forecasts
        
        # Si es un solo tensor, procesarlo directamente
        else:
            forecasts = self.model_loader.generate_forecasts(
                time_series_data,
                prediction_length=self.generation_params.get('max_length', 24)
            )
            
            logger.info(f"Generados pronósticos sintéticos para una serie temporal")
            return [forecasts]
    
    def _parse_generated_text(self, text: str) -> Dict:
        """
        Parsea el texto generado a un diccionario de valores
        
        Args:
            text: Texto generado por el modelo
            
        Returns:
            Diccionario con valores parseados
        """
        # Extraer pares clave-valor del texto generado
        # Este parser debe adaptarse al formato específico que use el modelo
        pairs = {}
        
        # Intenta extraer como si fuera un formato "key: value, key2: value2"
        pattern = r'([^,]+):\s*([^,]+)(?:,|$)'
        matches = re.findall(pattern, text)
        
        for key, value in matches:
            key = key.strip()
            value = value.strip()
            
            # Intentar convertir valores numéricos
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Mantener como string si no es numérico
            
            pairs[key] = value
        
        return pairs
    
    def generate_synthetic_dataset(self, time_series_data: List[torch.Tensor], 
                                   original_df: pd.DataFrame, 
                                   output_file: str = "synthetic_data.csv") -> pd.DataFrame:
        """
        Genera un conjunto de datos sintéticos completo a partir de series temporales
        
        Args:
            time_series_data: Lista de tensores con series temporales
            original_df: DataFrame original para referencia de estructura
            output_file: Nombre del archivo de salida
            
        Returns:
            DataFrame con datos sintéticos
        """
        logger.info(f"Generando dataset sintético a partir de {len(time_series_data)} series temporales")
        
        # Generar pronósticos con Chronos
        forecasts = self.generate(time_series_data)
        
        # Convertir pronósticos a DataFrame
        synthetic_data = []
        
        # Crear columnas para los datos sintéticos
        for i, forecast_batch in enumerate(forecasts):
            # forecast_batch tiene forma [num_samples, prediction_length]
            num_samples, pred_length = forecast_batch.shape[:2]
            
            # Para cada muestra en el lote
            for sample_idx in range(num_samples):
                sample_forecast = forecast_batch[sample_idx]
                
                # Crear un diccionario con la serie sintética
                row_dict = {
                    'series_id': f'synthetic_{i}_{sample_idx}',
                    'source_series': i,
                    'sample_id': sample_idx
                }
                
                # Añadir cada punto de la predicción como una columna
                for t in range(pred_length):
                    row_dict[f'value_t{t}'] = float(sample_forecast[t])
                
                synthetic_data.append(row_dict)
        
        # Crear DataFrame con los datos sintéticos
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Guardar datos sintéticos
        output_path = os.path.join(self.synthetic_path, output_file)
        synthetic_df.to_csv(output_path, index=False)
        logger.info(f"Datos sintéticos guardados en {output_path}")
        
        # También guardar cada serie individual en formato adecuado
        for i, forecast_batch in enumerate(forecasts):
            # Guardar la mediana (percentil 50) como la predicción principal
            median_forecast = np.median(forecast_batch, axis=0)
            
            # Calcular intervalos de predicción (10% y 90%)
            lower_bound = np.percentile(forecast_batch, 10, axis=0)
            upper_bound = np.percentile(forecast_batch, 90, axis=0)
            
            # Crear DataFrame con la predicción y sus intervalos
            forecast_df = pd.DataFrame({
                'time_idx': list(range(len(median_forecast))),
                'median_forecast': median_forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            # Guardar en archivo separado
            forecast_path = os.path.join(self.synthetic_path, f'forecast_series_{i}.csv')
            forecast_df.to_csv(forecast_path, index=False)
            logger.info(f"Pronóstico para serie {i} guardado en {forecast_path}")
        
        return synthetic_df


if __name__ == "__main__":
    # Prueba rápida del generador
    from src.data_preprocessing import DataPreprocessor
    
    try:
        # Cargar y preprocesar datos
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data("sample.csv")
        processed_df = preprocessor.preprocess_data(df)
        
        # Formatear para Chronos-T5
        formatted_texts = preprocessor.prepare_for_chronos(processed_df)
        
        # Generar datos sintéticos
        generator = SyntheticDataGenerator()
        synthetic_df = generator.generate_synthetic_dataset(
            formatted_texts[:10],  # Usar sólo 10 para prueba
            processed_df,
            "test_synthetic.csv"
        )
        
        print(f"Datos sintéticos generados con forma: {synthetic_df.shape}")
        print(synthetic_df.head(2))
        
    except FileNotFoundError:
        print("Archivo de ejemplo no encontrado. Por favor, coloca un archivo en la carpeta 'data/raw/'")
    except Exception as e:
        print(f"Error durante la prueba: {str(e)}")