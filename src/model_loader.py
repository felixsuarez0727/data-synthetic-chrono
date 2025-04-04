"""
Module for loading the Chronos-T5-Small model from Hugging Face
using the official Chronos pipeline
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
    Class for loading and managing the Chronos-T5 model using
    the official Chronos pipeline
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initializes the model loader with the specified configuration
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.model_name = self.config['model']['name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        self.pipeline = None
    
    def load_model(self):
        """
        Loads the Chronos pipeline
        
        Returns:
            tuple: (pipeline, None) - The second element is None to maintain compatibility
        """
        try:
            # Dynamically import ChronosPipeline
            try:
                from chronos import ChronosPipeline
            except ImportError:
                logger.error("Could not import ChronosPipeline. Make sure to install the package with:")
                logger.error("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
                raise
            
            logger.info(f"Loading model {self.model_name} with ChronosPipeline...")
            
            # Determine data type for loading the model
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
            
            # Load the Chronos pipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=dtype
            )
            
            logger.info("Chronos Pipeline loaded successfully")
            return self.pipeline, None  # Return None as second element to maintain compatibility
        
        except Exception as e:
            logger.error(f"Error loading the Chronos pipeline: {str(e)}")
            raise
    
    def get_generation_params(self):
        """
        Gets the generation parameters from the configuration
        
        Returns:
            dict: Generation parameters
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
        Generates forecasts using the Chronos pipeline
        
        Args:
            time_series_data: Time series data (tensor, list of tensors, or numpy array)
            prediction_length: Prediction length (if None, the value from config is used)
            
        Returns:
            numpy.ndarray: Generated forecasts
        """
        if self.pipeline is None:
            logger.error("Pipeline not initialized. Call load_model() first.")
            raise ValueError("Pipeline not initialized")
        
        if prediction_length is None:
            prediction_length = self.config['model']['max_length']
        
        # Convert data to tensor if necessary
        if not isinstance(time_series_data, torch.Tensor):
            time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
        
        # Get generation parameters
        gen_params = self.get_generation_params()
        
        # Generate forecasts
        logger.info(f"Generating forecasts for time series with prediction length {prediction_length}")
        
        try:
            # The ChronosPipeline API only accepts these specific parameters
            forecasts = self.pipeline.predict(
                time_series_data, 
                prediction_length,
                num_samples=gen_params['num_samples'],
                temperature=gen_params['temperature']
                # do_sample is not a valid parameter for ChronosPipeline.predict()
            )
            
            return forecasts.numpy() if isinstance(forecasts, torch.Tensor) else forecasts
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {str(e)}")
            raise


if __name__ == "__main__":
    # Quick test to verify that everything works correctly
    loader = ChronosModelLoader()
    pipeline, _ = loader.load_model()
    print(f"Pipeline loaded: {type(pipeline).__name__}")
    
    # Create test data
    test_data = torch.tensor(np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50))
    forecasts = loader.generate_forecasts(test_data, 10)
    print(f"Forecasts generated with shape: {forecasts.shape}")