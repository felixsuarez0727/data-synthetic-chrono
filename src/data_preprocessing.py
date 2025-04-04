"""
Módulo para preprocesar datos antes de generar datos sintéticos
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
    Clase para preprocesar datos para uso con Chronos-T5
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Inicializa el preprocesador con la configuración especificada
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_path = self.config['data']['raw_path']
        self.processed_path = self.config['data']['processed_path']
        
        # Crear directorios si no existen
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Carga datos desde el directorio de datos raw
        
        Args:
            filename: Nombre del archivo a cargar
            
        Returns:
            DataFrame con los datos cargados
        """
        file_path = os.path.join(self.raw_path, filename)
        logger.info(f"Cargando datos desde {file_path}")
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.json'):
            df = pd.read_json(file_path)
        elif filename.endswith('.hdf5') or filename.endswith('.h5'):
            # Cargar archivo HDF5
            import h5py
            
            # Convertir datos HDF5 a DataFrame
            with h5py.File(file_path, 'r') as f:
                # Mostrar estructura del archivo HDF5
                logger.info("Estructura del archivo HDF5:")
                
                def print_attrs(name, obj):
                    logger.info(f"- {name}: {type(obj)}")
                    return None
                
                f.visititems(print_attrs)
                
                # Identificar datasets dentro del archivo HDF5
                datasets = {}
                
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = obj
                    return None
                
                f.visititems(collect_datasets)
                
                # Crear un DataFrame con todos los datasets encontrados
                data_dict = {}
                for name, dataset in datasets.items():
                    # Convertir a array de numpy y luego a lista
                    try:
                        # Extraer datos, manejar diferentes dimensiones
                        if dataset.shape:  # Asegurarse que no esté vacío
                            if len(dataset.shape) == 1:  # Vector 1D
                                data_dict[name] = dataset[:]
                            elif len(dataset.shape) == 2:  # Matriz 2D
                                # Para cada columna en el dataset 2D
                                for i in range(dataset.shape[1]):
                                    data_dict[f"{name}_col{i}"] = dataset[:, i]
                            else:
                                # Para datasets multidimensionales, aplanar
                                flat_data = dataset[:].flatten()
                                data_dict[f"{name}_flattened"] = flat_data[:min(len(flat_data), 1000)]  # Limitar tamaño
                    except Exception as e:
                        logger.warning(f"No se pudieron extraer datos de {name}: {e}")
                
                # Crear DataFrame, asegurando que todas las columnas tengan la misma longitud
                max_len = max([len(arr) for arr in data_dict.values()]) if data_dict else 0
                
                for key, val in data_dict.items():
                    if len(val) < max_len:
                        # Rellenar con NaN si es necesario
                        data_dict[key] = np.pad(val, (0, max_len - len(val)), 
                                               'constant', constant_values=np.nan)
                
                df = pd.DataFrame(data_dict)
                
                # Si el DataFrame está vacío, intentar otra estrategia
                if df.empty:
                    logger.warning("No se pudieron extraer datos usando el método estándar. Intentando método alternativo.")
                    # Crear un DataFrame simple con algunos atributos del archivo
                    attrs_dict = {}
                    for name, dataset in datasets.items():
                        # Añadir atributos como metadatos
                        for attr_name, attr_value in dataset.attrs.items():
                            attrs_dict[f"{name}_{attr_name}"] = [str(attr_value)]
                    
                    if attrs_dict:
                        df = pd.DataFrame(attrs_dict)
                    else:
                        # Si no hay datos útiles, crear un DataFrame mínimo
                        df = pd.DataFrame({'filename': [filename], 'hdf5_keys': [', '.join(list(datasets.keys()))]})
        else:
            raise ValueError(f"Formato de archivo no soportado: {filename}")
        
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    def _detect_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detecta los tipos de datos en cada columna
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            Diccionario con tipo de datos para cada columna
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
        Preprocesa los datos para su uso con Chronos-T5
        
        Args:
            df: DataFrame con datos raw
            output_file: Nombre del archivo de salida
            
        Returns:
            DataFrame preprocesado
        """
        logger.info("Iniciando preprocesamiento de datos...")
        
        # Detectar tipos de columnas
        column_types = self._detect_data_types(df)
        logger.info(f"Tipos de columnas detectados: {column_types}")
        
        # Procesar valores faltantes
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Columna '{col}' tiene {missing_count} valores faltantes")
                
                if column_types[col] in ['integer', 'float']:
                    df[col] = df[col].fillna(df[col].median())
                elif column_types[col] == 'categorical':
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna("")
        
        # Convertir a formato adecuado para Chronos-T5
        # Chronos-T5 trabaja con texto, por lo que necesitamos convertir los datos
        # a un formato que el modelo pueda entender
        
        # Guardar el dataframe procesado
        output_path = os.path.join(self.processed_path, output_file)
        df.to_csv(output_path, index=False)
        logger.info(f"Datos preprocesados guardados en {output_path}")
        
        return df
    
    def prepare_for_chronos(self, df: pd.DataFrame) -> List[torch.Tensor]:
        """
        Prepara los datos para ser procesados por Chronos-T5
        
        Args:
            df: DataFrame preprocesado
            
        Returns:
            Lista de tensores para Chronos
        """
        # Chronos-T5 espera datos de series temporales como tensores
        # Necesitamos extraer las series temporales del DataFrame
        
        import torch
        
        # Identificar columnas numéricas (potencialmente series temporales)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            logger.warning("No se encontraron columnas numéricas. Chronos requiere datos numéricos de series temporales.")
            # Crear algunas series temporales sintéticas básicas como ejemplo
            series_data = [torch.tensor(np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100))]
            return series_data
        
        # Extraer series temporales
        series_data = []
        
        # Caso 1: Si hay muchas filas y pocas columnas numéricas, asumimos que cada columna es una serie temporal
        if len(df) > 10 and len(numeric_cols) < 10:
            for col in numeric_cols:
                series = df[col].dropna().values
                if len(series) > 10:  # Solo si hay suficientes puntos de datos
                    series_data.append(torch.tensor(series, dtype=torch.float32))
        
        # Caso 2: Si hay pocas filas y muchas columnas numéricas, asumimos que cada fila es un punto en varias series temporales
        elif len(df) < 10 and len(numeric_cols) > 10:
            # Transponer el DataFrame para obtener series temporales
            transposed_data = df[numeric_cols].T.values
            for i in range(min(10, transposed_data.shape[0])):  # Limitar a 10 series
                series_data.append(torch.tensor(transposed_data[i], dtype=torch.float32))
        
        # Caso 3: Si hay estructura mixta, intentar buscar patrones de series temporales
        else:
            # Buscar columnas que parezcan índices temporales
            potential_time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'timestamp' in col.lower()]
            
            if potential_time_cols:
                time_col = potential_time_cols[0]
                # Ordenar por la columna temporal
                df_sorted = df.sort_values(by=time_col)
                
                # Extraer series para cada columna numérica
                for col in numeric_cols:
                    if col != time_col:
                        series = df_sorted[col].dropna().values
                        if len(series) > 10:
                            series_data.append(torch.tensor(series, dtype=torch.float32))
            else:
                # Si no encontramos estructura clara, tomamos las primeras columnas numéricas
                for col in numeric_cols[:min(5, len(numeric_cols))]:
                    series = df[col].dropna().values
                    if len(series) > 10:
                        series_data.append(torch.tensor(series, dtype=torch.float32))
        
        # Si no encontramos series adecuadas, crear una serie sintética básica
        if not series_data:
            logger.warning("No se encontraron series temporales adecuadas. Usando una serie sintética de ejemplo.")
            series_data = [torch.tensor(np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100))]
        
        logger.info(f"Se han preparado {len(series_data)} series temporales para Chronos-T5")
        return series_data
    
    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba
        
        Args:
            df: DataFrame a dividir
            
        Returns:
            Tupla de (train_df, test_df)
        """
        test_size = self.config['evaluation']['test_size']
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.config['generation']['seed']
        )
        
        logger.info(f"Datos divididos: {train_df.shape[0]} para entrenamiento, {test_df.shape[0]} para pruebas")
        
        # Guardar los conjuntos
        train_df.to_csv(os.path.join(self.processed_path, "train_data.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_path, "test_data.csv"), index=False)
        
        return train_df, test_df


if __name__ == "__main__":
    # Prueba rápida del preprocesador
    preprocessor = DataPreprocessor()
    # Asumiendo que hay un archivo sample.csv en la carpeta raw
    try:
        df = preprocessor.load_data("sample.csv")
        processed_df = preprocessor.preprocess_data(df)
        train_df, test_df = preprocessor.split_train_test(processed_df)
        formatted_texts = preprocessor.prepare_for_chronos(train_df.head(5))
        print("Ejemplo de textos formateados:")
        for text in formatted_texts[:2]:
            print(text)
    except FileNotFoundError:
        print("Archivo de ejemplo no encontrado. Por favor, coloca un archivo en la carpeta 'data/raw/'")