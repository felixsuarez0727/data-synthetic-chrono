"""
Módulo para evaluar la calidad de los pronósticos de series temporales generados
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesEvaluator:
    """
    Clase para evaluar la calidad de los pronósticos de series temporales generados
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Inicializa el evaluador con la configuración especificada
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_path = self.config['data']['processed_path']
        self.synthetic_path = self.config['data']['synthetic_path']
        self.metrics = self.config['evaluation']['metrics']
        
        # Crear directorio para resultados de evaluación
        self.results_path = os.path.join(self.synthetic_path, "evaluation")
        os.makedirs(self.results_path, exist_ok=True)
    
    def load_datasets(self, real_data_file: str, synthetic_data_file: str) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """
        Carga los conjuntos de datos real y los pronósticos sintéticos
        
        Args:
            real_data_file: Nombre del archivo con datos reales
            synthetic_data_file: Nombre del archivo con datos sintéticos
            
        Returns:
            Tupla de (datos_reales, pronósticos)
        """
        # Cargar datos reales
        real_path = os.path.join(self.processed_path, real_data_file)
        logger.info(f"Cargando datos reales desde {real_path}")
        real_df = pd.read_csv(real_path)
        
        # Cargar pronósticos sintéticos
        synthetic_path = os.path.join(self.synthetic_path, synthetic_data_file)
        logger.info(f"Cargando datos sintéticos desde {synthetic_path}")
        synthetic_df = pd.read_csv(synthetic_path)
        
        # Cargar pronósticos individuales
        forecasts = {}
        forecast_files = [f for f in os.listdir(self.synthetic_path) if f.startswith('forecast_series_') and f.endswith('.csv')]
        
        for file in forecast_files:
            series_id = int(file.split('_')[-1].split('.')[0])
            forecast_path = os.path.join(self.synthetic_path, file)
            forecasts[series_id] = pd.read_csv(forecast_path)
        
        # Extraer series temporales del DataFrame real
        real_series = {}
        
        # Identificar columnas numéricas (potenciales series temporales)
        numeric_cols = real_df.select_dtypes(include=['number']).columns.tolist()
        
        # Método simple: usar cada columna numérica como una serie temporal
        for i, col in enumerate(numeric_cols):
            series = real_df[col].dropna().values
            if len(series) > 10:  # Solo si hay suficientes puntos de datos
                real_series[i] = series
        
        return real_series, forecasts
    
    def evaluate_forecast_accuracy(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evalúa la precisión de un pronóstico
        
        Args:
            actual: Valores reales
            forecast: Valores pronosticados
            
        Returns:
            Diccionario con métricas de precisión
        """
        # Si los tamaños no coinciden, recortar al más corto
        min_length = min(len(actual), len(forecast))
        actual = actual[-min_length:]
        forecast = forecast[:min_length]
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        
        # MAPE puede dar error si hay valores cero en actual
        try:
            mape = mean_absolute_percentage_error(actual, forecast) * 100
        except:
            mape = np.nan
        
        # Coeficiente de determinación (R²)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        ss_res = np.sum((actual - forecast) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def visualize_forecast(self, series_id: int, actual: np.ndarray, forecast_df: pd.DataFrame, prediction_length: int = 24) -> None:
        """
        Visualiza una serie temporal y su pronóstico
        
        Args:
            series_id: ID de la serie temporal
            actual: Valores reales
            forecast_df: DataFrame con los pronósticos
            prediction_length: Longitud del pronóstico
            
        Returns:
            None (guarda la visualización en un archivo)
        """
        plt.figure(figsize=(12, 6))
        
        # Datos históricos
        history_idx = range(len(actual) - prediction_length)
        history = actual[:-prediction_length] if len(actual) > prediction_length else actual
        plt.plot(history_idx, history, 'b-', label='Datos históricos')
        
        # Valores reales del período de pronóstico (si están disponibles)
        if len(actual) > prediction_length:
            actual_future = actual[-prediction_length:]
            future_idx = range(len(actual) - prediction_length, len(actual))
            plt.plot(future_idx, actual_future, 'k-', label='Valores reales')
        
        # Pronóstico
        if 'median_forecast' in forecast_df.columns:
            forecast_idx = range(len(actual) - prediction_length, len(actual) - prediction_length + len(forecast_df))
            plt.plot(forecast_idx, forecast_df['median_forecast'], 'r-', label='Pronóstico (mediana)')
            
            # Intervalos de predicción (si están disponibles)
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                plt.fill_between(
                    forecast_idx,
                    forecast_df['lower_bound'],
                    forecast_df['upper_bound'],
                    color='r', alpha=0.3,
                    label='Intervalo 80% de predicción'
                )
        
        # Ajustar gráfico
        plt.title(f'Pronóstico para serie temporal {series_id}')
        plt.grid(True)
        plt.legend()
        
        # Guardar visualización
        plt.savefig(os.path.join(self.results_path, f'forecast_series_{series_id}.png'))
        plt.close()
        
        logger.info(f"Visualización guardada para serie {series_id}")
    
    def evaluate_all_forecasts(self, real_series: Dict[str, np.ndarray], forecasts: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Evalúa todos los pronósticos disponibles
        
        Args:
            real_series: Diccionario con series temporales reales
            forecasts: Diccionario con DataFrames de pronósticos
            
        Returns:
            Diccionario con métricas para cada serie temporal
        """
        all_metrics = {}
        
        for series_id, forecast_df in forecasts.items():
            # Verificar si tenemos datos reales para esta serie
            if series_id not in real_series:
                logger.warning(f"No hay datos reales para la serie {series_id}")
                continue
            
            actual = real_series[series_id]
            predicted = forecast_df['median_forecast'].values
            
            # Calcular métricas
            metrics = self.evaluate_forecast_accuracy(actual, predicted)
            all_metrics[str(series_id)] = metrics
            
            # Visualizar pronóstico
            self.visualize_forecast(series_id, actual, forecast_df)
        
        return all_metrics
    
    def visualize_metric_distribution(self, all_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Visualiza la distribución de las métricas de evaluación
        
        Args:
            all_metrics: Diccionario con métricas para cada serie temporal
            
        Returns:
            None (guarda la visualización en un archivo)
        """
        # Convertir diccionario de métricas a DataFrame
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        
        # Visualizar distribución de métricas
        plt.figure(figsize=(15, 10))
        
        # RMSE
        plt.subplot(2, 2, 1)
        sns.histplot(metrics_df['rmse'].dropna())
        plt.title('Distribución de RMSE')
        plt.xlabel('RMSE')
        
        # MAE
        plt.subplot(2, 2, 2)
        sns.histplot(metrics_df['mae'].dropna())
        plt.title('Distribución de MAE')
        plt.xlabel('MAE')
        
        # MAPE
        plt.subplot(2, 2, 3)
        sns.histplot(metrics_df['mape'].dropna())
        plt.title('Distribución de MAPE (%)')
        plt.xlabel('MAPE (%)')
        
        # R²
        plt.subplot(2, 2, 4)
        sns.histplot(metrics_df['r2'].dropna())
        plt.title('Distribución de R²')
        plt.xlabel('R²')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'metric_distributions.png'))
        plt.close()
        
        logger.info("Visualización de distribución de métricas guardada")
    
    def run_evaluation(self, real_data_file: str = "test_data.csv", 
                      synthetic_data_file: str = "synthetic_data.csv") -> Dict:
        """
        Ejecuta la evaluación completa de pronósticos
        
        Args:
            real_data_file: Nombre del archivo con datos reales
            synthetic_data_file: Nombre del archivo con datos sintéticos
            
        Returns:
            Diccionario con resultados de evaluación
        """
        logger.info("Iniciando evaluación de pronósticos de series temporales...")
        
        # Cargar datos
        real_series, forecasts = self.load_datasets(real_data_file, synthetic_data_file)
        
        # Si no hay series para evaluar
        if not real_series or not forecasts:
            logger.warning("No hay suficientes datos para evaluar")
            return {}
        
        # Evaluar todos los pronósticos
        all_metrics = self.evaluate_all_forecasts(real_series, forecasts)
        
        # Visualizar distribución de métricas
        if all_metrics:
            self.visualize_metric_distribution(all_metrics)
        
        # Calcular métricas promedio
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            values = [m[metric] for m in all_metrics.values() if not np.isnan(m[metric])]
            avg_metrics[f'avg_{metric}'] = np.mean(values) if values else np.nan
        
        # Combinar resultados
        results = {
            'individual_metrics': all_metrics,
            'average_metrics': avg_metrics
        }
        
        # Guardar resultados
        results_file = os.path.join(self.results_path, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            import json
            # Convertir valores numpy a Python nativos para serialización JSON
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Crear versión serializable
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: {sk: convert_numpy(sv) for sk, sv in v.items()} if isinstance(v, dict) else convert_numpy(v) for k, v in value.items()}
                else:
                    serializable_results[key] = convert_numpy(value)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluación completada. Resultados guardados en {results_file}")
        
        return results


if __name__ == "__main__":
    # Prueba rápida del evaluador
    try:
        evaluator = TimeSeriesEvaluator()
        results = evaluator.run_evaluation()
        print("Evaluación completada con éxito")
    except FileNotFoundError:
        print("Archivos de datos no encontrados. Primero genera pronósticos sintéticos.")
    except Exception as e:
        print(f"Error durante la evaluación: {str(e)}")