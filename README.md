# Synthetic Time Series Generation with Chronos-T5

This project implements a comprehensive workflow for generating synthetic time series data using Amazon's Chronos-T5 model. It provides tools for data preprocessing, time series extraction, synthetic forecast generation, and quality evaluation.

## Project Overview

Chronos-T5 is a family of pretrained time series forecasting models based on language model architectures. This project leverages the official Chronos pipeline to generate probabilistic forecasts that can be used as synthetic time series data.

The key features of this project include:
- Loading and preprocessing time series data from various formats (CSV, Excel, JSON, HDF5)
- Extracting time series patterns from complex datasets
- Generating synthetic forecasts using Chronos-T5
- Evaluating forecast quality with standard metrics (RMSE, MAE, MAPE, R²)
- Visualizing both real and synthetic time series data

## Project Structure

```
chronos-synthetic-data/
│
├── README.md                      # Project documentation
├── requirements.txt               # Project dependencies
├── config/
│   └── config.yaml                # Project configuration
│
├── data/
│   ├── raw/                       # Original data files
│   ├── processed/                 # Processed data
│   └── synthetic/                 # Generated synthetic data
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data preprocessing module
│   ├── model_loader.py            # Chronos-T5 model loader
│   ├── synthetic_generator.py     # Synthetic data generator
│   └── evaluation.py              # Evaluation module
│
├── notebooks/
│   ├── exploratory_analysis.ipynb # Exploratory data analysis
│   └── model_evaluation.ipynb     # Model evaluation
│
└── main.py                        # Main execution script
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/chronos-synthetic-data.git
cd chronos-synthetic-data
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Chronos pipeline:
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

## Usage

### 1. Data Preparation

Place your time series data file in the `data/raw/` directory. The project supports CSV, Excel, JSON, and HDF5 formats.

### 2. Running the Main Script

Execute the main script with your input file:

```bash
python main.py --input your_file.hdf5 --output synthetic_data.csv
```

Options:
- `--input` or `-i`: Input data file in the `data/raw/` directory (required)
- `--output` or `-o`: Name for the output file (optional)
- `--config` or `-c`: Custom configuration file path (optional)

### 3. Exploring and Evaluating Results

After running the main script, you can:
- Check the generated synthetic data in `data/synthetic/`
- View evaluation results in `data/synthetic/evaluation/`
- Use the provided notebooks for in-depth analysis:
  - `notebooks/exploratory_analysis.ipynb`: For exploring the input data
  - `notebooks/model_evaluation.ipynb`: For evaluating the generated forecasts

## Configuration

You can customize the project's behavior by modifying the `config/config.yaml` file. Key parameters include:

- `model.name`: Chronos model to use (default: "amazon/chronos-t5-small")
- `model.max_length`: Forecast horizon (number of time steps to predict)
- `model.num_beams`: Number of forecast trajectories to sample
- `model.temperature`: Controls randomness in generation (higher = more diverse)

## Example

Here's a minimal example to test the project:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from chronos import ChronosPipeline

# Create sample time series
t = np.linspace(0, 4, 200)
series = 0.5 * np.sin(2 * np.pi * t * 10) + 0.8 * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))
time_series = torch.tensor(series, dtype=torch.float32)

# Load Chronos-T5 model
pipeline = ChronosPipeline.from_pretrained(
  "amazon/chronos-t5-small",
  device_map="auto",
  torch_dtype=torch.float32
)

# Generate forecast
prediction_length = 48
forecast = pipeline.predict(time_series, prediction_length, num_samples=10, temperature=0.8)

# Calculate forecast statistics
forecast_np = forecast.numpy()
median = np.median(forecast_np, axis=0)
lower = np.percentile(forecast_np, 10, axis=0)
upper = np.percentile(forecast_np, 90, axis=0)

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(series, color="blue", label="historical data")
plt.plot(range(len(series), len(series) + prediction_length), median, color="red", label="median forecast")
plt.fill_between(range(len(series), len(series) + prediction_length), lower, upper, 
                color="red", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()
```

## Troubleshooting

If you encounter any issues:

1. **GPU Memory Errors**: For large models or datasets, try using CPU by setting `device_map="cpu"` in config.yaml.

2. **HDF5 Import Errors**: Ensure h5py is installed correctly: `pip install h5py`.

3. **Time Series Extraction**: If no time series are found, check your data structure and consider modifying `prepare_for_chronos()` in `data_preprocessing.py`.

4. **Model Download Issues**: Ensure you have a stable internet connection when first running the model, as it will download weights from Hugging Face.

## References

- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [Official Chronos Repository](https://github.com/amazon-science/chronos-forecasting)
- [Hugging Face Model Page](https://huggingface.co/amazon/chronos-t5-small)

