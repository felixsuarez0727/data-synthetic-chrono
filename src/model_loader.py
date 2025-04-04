"""
Módulo para cargar el modelo Chronos-T5-Small desde Hugging Face
utilizando el pipeline oficial de Chronos
"""

import torch
import logging
import yaml
import numpy as np
from typing import Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChronosModelLoader:
    """
    Clase para cargar y gestionar el modelo Chronos-T5 utilizando
    el pipeline oficial de Chronos
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Inicializa el cargador de modelo con la configuración especificada
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.model_name = self.config['model']['name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Utilizando dispositivo: {self.device}")
        
        self.pipeline = None
    
    def load_model(self):
        """
        Carga el pipeline de Chronos
        
        Returns:
            tuple: (pipeline, None) - El segundo elemento es None para mantener compatibilidad
        """
        try:
            # Importar ChronosPipeline de forma dinámica
            try:
                from chronos import ChronosPipeline
            except ImportError:
                logger.error("No se pudo importar ChronosPipeline. Asegúrate de instalar el paquete con:")
                logger.error("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
                raise
            
            logger.info(f"Cargando modelo {self.model_name} con ChronosPipeline...")
            
            # Determinar tipo de datos para cargar el modelo
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
            
            # Cargar el pipeline de Chronos
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=dtype
            )
            
            logger.info("Pipeline de Chronos cargado correctamente")
            return self.pipeline, None  # Retornamos None como segundo elemento para mantener compatibilidad
        
        except Exception as e:
            logger.error(f"Error al cargar el pipeline de Chronos: {str(e)}")
            raise
    
    def get_generation_params(self):
        """
        Obtiene los parámetros de generación desde la configuración
        
        Returns:
            dict: Parámetros de generación
        """
        gen_params = {
            'num_samples': self.config['model']['num_beams'],
            'temperature': self.config['model']['temperature'],
            'top_k': self.config['model']['top_k'],
            'top_p': self.config['model']['top_p'],
            'do_sample': self.config['model']['do_sample']
        }
        return gen_params
    
    def generate_forecasts(self, time_series_data, prediction_length=None):
        """
        Genera pronósticos utilizando el pipeline de Chronos
        
        Args:
            time_series_data: Datos de series temporales (tensor, lista de tensores, o numpy array)
            prediction_length: Longitud de la predicción (si es None, se usa el valor de config)
            
        Returns:
            numpy.ndarray: Pronósticos generados
        """
        if self.pipeline is None:
            logger.error("Pipeline no inicializado. Llama a load_model() primero.")
            raise ValueError("Pipeline no inicializado")
        
        if prediction_length is None:
            prediction_length = self.config['model']['max_length']
        
        # Convertir datos a tensor si es necesario
        if not isinstance(time_series_data, torch.Tensor):
            time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
        
        # Obtener parámetros de generación
        gen_params = self.get_generation_params()
        
        # Generar pronósticos
        logger.info(f"Generando pronósticos para series temporales con longitud de predicción {prediction_length}")
        
        try:
            # La API de ChronosPipeline solo acepta estos parámetros específicos
            forecasts = self.pipeline.predict(
                time_series_data, 
                prediction_length,
                num_samples=gen_params['num_samples'],
                temperature=gen_params['temperature']
                # do_sample no es un parámetro válido para ChronosPipeline.predict()
            )
            
            return forecasts.numpy() if isinstance(forecasts, torch.Tensor) else forecasts
            
        except Exception as e:
            logger.error(f"Error al generar pronósticos: {str(e)}")
            raise


if __name__ == "__main__":
    # Prueba rápida para verificar que todo funciona correctamente
    loader = ChronosModelLoader()
    pipeline, _ = loader.load_model()
    print(f"Pipeline cargado: {type(pipeline).__name__}")
    
    # Crear datos de prueba
    test_data = torch.tensor(np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50))
    forecasts = loader.generate_forecasts(test_data, 10)
    print(f"Pronósticos generados con forma: {forecasts.shape}")