"""
Module for evaluating the quality of generated time series forecasts
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
    Class for evaluating the quality of generated time series forecasts
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initializes the evaluator with the specified configuration
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_path = self.config['data']['processed_path']
        self.synthetic_path = self.config['data']['synthetic_path']
        self.metrics = self.config['evaluation']['metrics']
        
        # Create directory for evaluation results
        self.results_path = os.path.join(self.synthetic_path, "evaluation")
        os.makedirs(self.results_path, exist_ok=True)
    
    def load_datasets(self, real_data_file: str, synthetic_data_file: str) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """
        Loads the real data sets and synthetic forecasts
        
        Args:
            real_data_file: Name of the file with real data
            synthetic_data_file: Name of the file with synthetic data
            
        Returns:
            Tuple of (real_data, forecasts)
        """
        # Load real data
        real_path = os.path.join(self.processed_path, real_data_file)
        logger.info(f"Loading real data from {real_path}")
        real_df = pd.read_csv(real_path)
        
        # Load synthetic forecasts
        synthetic_path = os.path.join(self.synthetic_path, synthetic_data_file)
        logger.info(f"Loading synthetic data from {synthetic_path}")
        synthetic_df = pd.read_csv(synthetic_path)
        
        # Load individual forecasts
        forecasts = {}
        forecast_files = [f for f in os.listdir(self.synthetic_path) if f.startswith('forecast_series_') and f.endswith('.csv')]
        
        for file in forecast_files:
            series_id = int(file.split('_')[-1].split('.')[0])
            forecast_path = os.path.join(self.synthetic_path, file)
            forecasts[series_id] = pd.read_csv(forecast_path)
        
        # Extract time series from the real DataFrame
        real_series = {}
        
        # Identify numeric columns (potential time series)
        numeric_cols = real_df.select_dtypes(include=['number']).columns.tolist()
        
        # Simple method: use each numeric column as a time series
        for i, col in enumerate(numeric_cols):
            series = real_df[col].dropna().values
            if len(series) > 10:  # Only if there are enough data points
                real_series[i] = series
        
        return real_series, forecasts
    
    def evaluate_forecast_accuracy(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the accuracy of a forecast
        
        Args:
            actual: Real values
            forecast: Forecasted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        # If sizes don't match, trim to the shortest one
        min_length = min(len(actual), len(forecast))
        actual = actual[-min_length:]
        forecast = forecast[:min_length]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        mae = mean_absolute_error(actual, forecast)
        
        # MAPE can error if there are zero values in actual
        try:
            mape = mean_absolute_percentage_error(actual, forecast) * 100
        except:
            mape = np.nan
        
        # Coefficient of determination (R²)
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
        Visualizes a time series and its forecast
        
        Args:
            series_id: ID of the time series
            actual: Real values
            forecast_df: DataFrame with the forecasts
            prediction_length: Length of the forecast
            
        Returns:
            None (saves the visualization to a file)
        """
        plt.figure(figsize=(12, 6))
        
        # Historical data
        history_idx = range(len(actual) - prediction_length)
        history = actual[:-prediction_length] if len(actual) > prediction_length else actual
        plt.plot(history_idx, history, 'b-', label='Historical data')
        
        # Real values of the forecast period (if available)
        if len(actual) > prediction_length:
            actual_future = actual[-prediction_length:]
            future_idx = range(len(actual) - prediction_length, len(actual))
            plt.plot(future_idx, actual_future, 'k-', label='Real values')
        
        # Forecast
        if 'median_forecast' in forecast_df.columns:
            forecast_idx = range(len(actual) - prediction_length, len(actual) - prediction_length + len(forecast_df))
            plt.plot(forecast_idx, forecast_df['median_forecast'], 'r-', label='Forecast (median)')
            
            # Prediction intervals (if available)
            if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
                plt.fill_between(
                    forecast_idx,
                    forecast_df['lower_bound'],
                    forecast_df['upper_bound'],
                    color='r', alpha=0.3,
                    label='80% prediction interval'
                )
        
        # Adjust plot
        plt.title(f'Forecast for time series {series_id}')
        plt.grid(True)
        plt.legend()
        
        # Save visualization
        plt.savefig(os.path.join(self.results_path, f'forecast_series_{series_id}.png'))
        plt.close()
        
        logger.info(f"Visualization saved for series {series_id}")
    
    def evaluate_all_forecasts(self, real_series: Dict[str, np.ndarray], forecasts: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Evaluates all available forecasts
        
        Args:
            real_series: Dictionary with real time series
            forecasts: Dictionary with forecast DataFrames
            
        Returns:
            Dictionary with metrics for each time series
        """
        all_metrics = {}
        
        for series_id, forecast_df in forecasts.items():
            # Check if we have real data for this series
            if series_id not in real_series:
                logger.warning(f"No real data available for series {series_id}")
                continue
            
            actual = real_series[series_id]
            predicted = forecast_df['median_forecast'].values
            
            # Calculate metrics
            metrics = self.evaluate_forecast_accuracy(actual, predicted)
            all_metrics[str(series_id)] = metrics
            
            # Visualize forecast
            self.visualize_forecast(series_id, actual, forecast_df)
        
        return all_metrics
    
    def visualize_metric_distribution(self, all_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Visualizes the distribution of evaluation metrics
        
        Args:
            all_metrics: Dictionary with metrics for each time series
            
        Returns:
            None (saves the visualization to a file)
        """
        # Convert metrics dictionary to DataFrame
        metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
        
        # Visualize metrics distribution
        plt.figure(figsize=(15, 10))
        
        # RMSE
        plt.subplot(2, 2, 1)
        sns.histplot(metrics_df['rmse'].dropna())
        plt.title('RMSE Distribution')
        plt.xlabel('RMSE')
        
        # MAE
        plt.subplot(2, 2, 2)
        sns.histplot(metrics_df['mae'].dropna())
        plt.title('MAE Distribution')
        plt.xlabel('MAE')
        
        # MAPE
        plt.subplot(2, 2, 3)
        sns.histplot(metrics_df['mape'].dropna())
        plt.title('MAPE (%) Distribution')
        plt.xlabel('MAPE (%)')
        
        # R²
        plt.subplot(2, 2, 4)
        sns.histplot(metrics_df['r2'].dropna())
        plt.title('R² Distribution')
        plt.xlabel('R²')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'metric_distributions.png'))
        plt.close()
        
        logger.info("Metrics distribution visualization saved")
    
    def run_evaluation(self, real_data_file: str = "test_data.csv", 
                      synthetic_data_file: str = "synthetic_data.csv") -> Dict:
        """
        Runs complete forecast evaluation
        
        Args:
            real_data_file: Name of the file with real data
            synthetic_data_file: Name of the file with synthetic data
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting time series forecast evaluation...")
        
        # Load data
        real_series, forecasts = self.load_datasets(real_data_file, synthetic_data_file)
        
        # If there are no series to evaluate
        if not real_series or not forecasts:
            logger.warning("Not enough data to evaluate")
            return {}
        
        # Evaluate all forecasts
        all_metrics = self.evaluate_all_forecasts(real_series, forecasts)
        
        # Visualize metrics distribution
        if all_metrics:
            self.visualize_metric_distribution(all_metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            values = [m[metric] for m in all_metrics.values() if not np.isnan(m[metric])]
            avg_metrics[f'avg_{metric}'] = np.mean(values) if values else np.nan
        
        # Combine results
        results = {
            'individual_metrics': all_metrics,
            'average_metrics': avg_metrics
        }
        
        # Save results
        results_file = os.path.join(self.results_path, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            import json
            # Convert numpy values to native Python for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Create serializable version
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: {sk: convert_numpy(sv) for sk, sv in v.items()} if isinstance(v, dict) else convert_numpy(v) for k, v in value.items()}
                else:
                    serializable_results[key] = convert_numpy(value)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved in {results_file}")
        
        return results


if __name__ == "__main__":
    # Quick test of the evaluator
    try:
        evaluator = TimeSeriesEvaluator()
        results = evaluator.run_evaluation()
        print("Evaluation completed successfully")
    except FileNotFoundError:
        print("Data files not found. First generate synthetic forecasts.")
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")