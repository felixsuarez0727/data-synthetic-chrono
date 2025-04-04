"""
Main entry point for the synthetic data generation project
with Chronos-T5 for time series
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
    """Sets up the necessary directories"""
    os.makedirs(config['data']['raw_path'], exist_ok=True)
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    os.makedirs(config['data']['synthetic_path'], exist_ok=True)

def load_config(config_path="config/config.yaml"):
    """Loads the configuration from the YAML file"""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def generate_synthetic_data(input_file, output_file=None, config_path="config/config.yaml"):
    """
    Generates synthetic data from an input file
    
    Args:
        input_file: Name of the input file in data/raw/
        output_file: Name of the output file (if None, it's generated automatically)
        config_path: Path to the configuration file
    """
    start_time = datetime.now()
    logger.info(f"Starting the synthetic data generation process using Chronos-T5 from {input_file}")
    
    # Load configuration
    config = load_config(config_path)
    setup_directories(config)
    
    # If no output file is specified, generate a name
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"synthetic_{timestamp}.csv"
    
    # 1. Preprocessing
    logger.info("Step 1: Preprocessing time series data")
    preprocessor = DataPreprocessor(config_path)
    
    try:
        # Load data
        df = preprocessor.load_data(input_file)
        processed_df = preprocessor.preprocess_data(df)
        train_df, test_df = preprocessor.split_train_test(processed_df)
        
        # 2. Prepare time series for Chronos-T5
        logger.info("Step 2: Preparing time series for Chronos-T5")
        time_series_data = preprocessor.prepare_for_chronos(train_df)
        
        # Verify that we have time series data
        if not time_series_data or len(time_series_data) == 0:
            raise ValueError("Could not extract time series from the provided data")
        
        logger.info(f"{len(time_series_data)} time series have been prepared")
        
        # Show information about the series
        for i, series in enumerate(time_series_data):
            logger.info(f"  Series {i}: {len(series)} points, range: [{series.min().item():.2f}, {series.max().item():.2f}]")
        
        # 3. Generation of synthetic forecasts
        logger.info("Step 3: Generating synthetic forecasts with Chronos-T5")
        generator = SyntheticDataGenerator(config_path)
        synthetic_df = generator.generate_synthetic_dataset(
            time_series_data,
            processed_df,
            output_file
        )
        
        # 4. Evaluation of forecast quality
        logger.info("Step 4: Evaluating forecast quality")
        evaluator = TimeSeriesEvaluator(config_path)
        evaluation_results = evaluator.run_evaluation("test_data.csv", output_file)
        
        # Show evaluation results
        if 'average_metrics' in evaluation_results:
            metrics = evaluation_results['average_metrics']
            logger.info("Evaluation results:")
            if 'avg_rmse' in metrics:
                logger.info(f"  Average RMSE: {metrics['avg_rmse']:.4f}")
            if 'avg_mae' in metrics:
                logger.info(f"  Average MAE: {metrics['avg_mae']:.4f}")
            if 'avg_mape' in metrics:
                logger.info(f"  Average MAPE: {metrics['avg_mape']:.2f}%")
            if 'avg_r2' in metrics:
                logger.info(f"  Average R²: {metrics['avg_r2']:.4f}")
        
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds() / 60
        logger.info(f"Process completed in {elapsed_time:.2f} minutes")
        logger.info(f"Synthetic data saved to data/synthetic/{output_file}")
        logger.info(f"Visualizations and evaluations saved to data/synthetic/evaluation/")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during the process: {str(e)}")
        raise
        
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Synthetic data generator with Chronos-T5')
    parser.add_argument('--input', '-i', required=True, help='Input data file (in data/raw/)')
    parser.add_argument('--output', '-o', help='Name of the output file (optional)')
    parser.add_argument('--config', '-c', default='config/config.yaml', help='Path to the configuration file')
    
    args = parser.parse_args()
    
    try:
        generate_synthetic_data(args.input, args.output, args.config)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())