"""
Module for preprocessing data before generating synthetic data
"""

import os
import pandas as pd
import numpy as np
import logging
import yaml
import torch
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing data for use with Chronos-T5
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initializes the preprocessor with the specified configuration
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_path = self.config['data']['raw_path']
        self.processed_path = self.config['data']['processed_path']
        
        # Create directories if they don't exist
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Loads data from the raw data directory
        
        Args:
            filename: Name of the file to load
            
        Returns:
            DataFrame with the loaded data
        """
        file_path = os.path.join(self.raw_path, filename)
        logger.info(f"Loading data from {file_path}")
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        elif filename.endswith('.hdf5') or filename.endswith('.h5'):
            # Load HDF5 file
            import h5py
            
            # Convert HDF5 data to DataFrame
            with h5py.File(file_path, 'r') as f:
                # Show HDF5 file structure
                logger.info("HDF5 file structure:")
                
                def print_attrs(name, obj):
                    logger.info(f"- {name}: {type(obj)}")
                    return None
                
                f.visititems(print_attrs)
                
                # Identify datasets within the HDF5 file
                datasets = {}
                
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = obj
                    return None
                
                f.visititems(collect_datasets)
                
                # Create a DataFrame with all found datasets
                data_dict = {}
                for name, dataset in datasets.items():
                    # Convert to numpy array and then to list
                    try:
                        # Extract data, handle different dimensions
                        if dataset.shape:  # Make sure it's not empty
                            if len(dataset.shape) == 1:  # 1D vector
                                data_dict[name] = dataset[:]
                            elif len(dataset.shape) == 2:  # 2D matrix
                                # For each column in the 2D dataset
                                for i in range(dataset.shape[1]):
                                    data_dict[f"{name}_col{i}"] = dataset[:, i]
                            else:
                                # For multidimensional datasets, flatten
                                flat_data = dataset[:].flatten()
                                data_dict[f"{name}_flattened"] = flat_data[:min(len(flat_data), 1000)]  # Limit size
                    except Exception as e:
                        logger.warning(f"Could not extract data from {name}: {e}")
                
                # Create DataFrame, ensuring all columns have the same length
                max_len = max([len(arr) for arr in data_dict.values()]) if data_dict else 0
                
                for key, val in data_dict.items():
                    if len(val) < max_len:
                        # Fill with NaN if necessary
                        data_dict[key] = np.pad(val, (0, max_len - len(val)), 
                                               'constant', constant_values=np.nan)
                
                df = pd.DataFrame(data_dict)
                
                # If the DataFrame is empty, try another strategy
                if df.empty:
                    logger.warning("Could not extract data using the standard method. Trying alternative method.")
                    # Create a simple DataFrame with some file attributes
                    attrs_dict = {}
                    for name, dataset in datasets.items():
                        # Add attributes as metadata
                        for attr_name, attr_value in dataset.attrs.items():
                            attrs_dict[f"{name}_{attr_name}"] = [str(attr_value)]
                    
                    if attrs_dict:
                        df = pd.DataFrame(attrs_dict)
                    else:
                        # If there's no useful data, create a minimal DataFrame
                        df = pd.DataFrame({'filename': [filename], 'hdf5_keys': [', '.join(list(datasets.keys()))]})
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detects data types in each column
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with data type for each column
        """
        column_types = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].apply(lambda x: x.is_integer() if not pd.isna(x) else True).all():
                    column_types[col] = 'integer'
                else:
                    column_types[col] = 'float'
            elif pd.api.types.is_datetime64_dtype(df[col]):
                column_types[col] = 'datetime'
            elif df[col].nunique() < 10 and df[col].nunique() / len(df[col]) < 0.1:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
        
        return column_types
    
    def preprocess_data(self, df: pd.DataFrame, output_file: str = "processed_data.csv") -> pd.DataFrame:
        """
        Preprocesses the data for use with Chronos-T5
        
        Args:
            df: DataFrame with raw data
            output_file: Name of the output file
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Detect column types
        column_types = self._detect_data_types(df)
        logger.info(f"Detected column types: {column_types}")
        
        # Process missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Column '{col}' has {missing_count} missing values")
                
                if column_types[col] in ['integer', 'float']:
                    df[col] = df[col].fillna(df[col].median())
                elif column_types[col] == 'categorical':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna("")
        
        # Convert to appropriate format for Chronos-T5
        # Chronos-T5 works with text, so we need to convert the data
        # to a format that the model can understand
        
        # Save the processed dataframe
        output_path = os.path.join(self.processed_path, output_file)
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")
        
        return df
    
    def prepare_for_chronos(self, df: pd.DataFrame) -> List[torch.Tensor]:
        """
        Prepares the data to be processed by Chronos-T5
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            List of tensors for Chronos
        """
        # Chronos-T5 expects time series data as tensors
        # We need to extract time series from the DataFrame
        
        import torch
        
        # Identify numeric columns (potentially time series)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No numeric columns found. Chronos requires numeric time series data.")
            # Create some basic synthetic time series as an example
            series_data = [torch.tensor(np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100))]
            return series_data
        
        # Extract time series
        series_data = []
        
        # Case 1: If there are many rows and few numeric columns, we assume each column is a time series
        if len(df) > 10 and len(numeric_cols) < 10:
            for col in numeric_cols:
                series = df[col].dropna().values
                if len(series) > 10:  # Only if there are enough data points
                    series_data.append(torch.tensor(series, dtype=torch.float32))
        
        # Case 2: If there are few rows and many numeric columns, we assume each row is a point in multiple time series
        elif len(df) < 10 and len(numeric_cols) > 10:
            # Transpose the DataFrame to get time series
            transposed_data = df[numeric_cols].T.values
            for i in range(min(10, transposed_data.shape[0])):  # Limit to 10 series
                series_data.append(torch.tensor(transposed_data[i], dtype=torch.float32))
        
        # Case 3: If there's a mixed structure, try to look for time series patterns
        else:
            # Look for columns that appear to be time indices
            potential_time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower()]
            
            if potential_time_cols:
                time_col = potential_time_cols[0]
                # Sort by the time column
                df_sorted = df.sort_values(by=time_col)
                
                # Extract series for each numeric column
                for col in numeric_cols:
                    if col != time_col:
                        series = df_sorted[col].dropna().values
                        if len(series) > 10:
                            series_data.append(torch.tensor(series, dtype=torch.float32))
            else:
                # If we don't find a clear structure, take the first numeric columns
                for col in numeric_cols[:min(5, len(numeric_cols))]:
                    series = df[col].dropna().values
                    if len(series) > 10:
                        series_data.append(torch.tensor(series, dtype=torch.float32))
        
        # If we didn't find suitable series, create a basic synthetic series
        if not series_data:
            logger.warning("No suitable time series found. Using a synthetic example series.")
            series_data = [torch.tensor(np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100))]
        
        logger.info(f"{len(series_data)} time series have been prepared for Chronos-T5")
        return series_data
    
    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and test sets
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (train_df, test_df)
        """
        test_size = self.config['evaluation']['test_size']
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.config['generation']['seed']
        )
        
        logger.info(f"Data split: {train_df.shape[0]} for training, {test_df.shape[0]} for testing")
        
        # Save the sets
        train_df.to_csv(os.path.join(self.processed_path, "train_data.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test_data.csv"), index=False)
        
        return train_df, test_df


if __name__ == "__main__":
    # Quick test of the preprocessor
    preprocessor = DataPreprocessor()
    # Assuming there's a sample.csv file in the raw folder
    try:
        df = preprocessor.load_data("sample.csv")
        processed_df = preprocessor.preprocess_data(df)
        train_df, test_df = preprocessor.split_train_test(processed_df)
        formatted_texts = preprocessor.prepare_for_chronos(train_df.head(5))
        print("Example of formatted texts:")
        for text in formatted_texts[:2]:
            print(text)
    except FileNotFoundError:
        print("Example file not found. Please place a file in the 'data/raw/' folder")