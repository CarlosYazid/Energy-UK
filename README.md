# Energy-UK: Solar Energy Forecasting Models

![Energy Forecast](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML Framework](https://img.shields.io/badge/TensorFlow-2.17.1-orange)
![Deployment](https://img.shields.io/badge/AzureML-Compatible-green)

## 1. Project Name
**Energy-UK** - Time Series Models for Solar Energy Consumption and Generation Forecasting in the United Kingdom

## 2. Brief Description
Energy-UK is a comprehensive machine learning project that provides accurate forecasts of solar energy consumption and generation across the United Kingdom. The system utilizes advanced time series models (LSTM and GRU networks) trained on historical energy data and weather patterns to predict:
- National electricity demand (`nd`)
- Solar energy generation (`solarenergy`)

The project includes complete ML pipelines from data preparation to model deployment on Azure ML, with integrated monitoring through MLflow.

## 3. Main Features

### Core Capabilities
- **Dual Forecasting Models**:
  - LSTM network for energy consumption prediction
  - GRU network for solar generation forecasting
- **Multi-modal Data Integration**:
  - Energy grid operational data
  - Meteorological measurements (solar radiation, cloud cover, etc.)
- **Production-Ready Infrastructure**:
  - MLflow experiment tracking
  - Azure ML deployment pipelines
  - Scalable inference endpoints

### Technical Highlights
- Automated feature engineering pipelines
- Dynamic windowing for time series data
- Hyperparameter tracking with MLflow
- Model versioning and registry
- CI/CD-ready deployment scripts

## 4. Prerequisites

### Hardware Requirements
- GPU-enabled system recommended (CUDA 12.x compatible)
- Minimum 16GB RAM
- 10GB free disk space

### Software Dependencies
```plaintext
Python 3.10+
TensorFlow 2.17.1
Azure ML SDK
MLflow 2.11.3
```

Complete dependency list available in:
- `requirements.txt` (1,200+ packages)
- Conda environment specifications

### Data Requirements
- UK National Grid historical data (provided in `uk_electricity.csv`)
- Meteorological records (provided in `uk_weather.csv`)

## 5. Installation

### Local Development Setup
1. Clone repository:
   ```bash
   git clone https://github.com/CarlosYazid/Energy-UK.git
   cd Energy-UK
   ```

2. Create Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate uk-energy-env
   ```

3. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Azure ML Workspace Configuration
1. Set environment variables:
   ```bash
   export AZURE_SUBSCRIPTION_ID=<your-subscription-id>
   export AZURE_RESOURCE_GROUP=<your-resource-group>
   export AZURE_WORKSPACE_NAME=<workspace-name>
   ```

2. Authenticate using Azure CLI:
   ```bash
   az login
   az account set --subscription $AZURE_SUBSCRIPTION_ID
   ```

## 6. Usage

### Model Training
**Energy Consumption Model**:
```bash
jupyter nbconvert --to python "Training-Energy-Comsumption.ipynb" && python Training-Energy-Comsumption.py
```

**Solar Generation Model**:
```bash
jupyter nbconvert --to python "Training-Generation-Energy.ipynb" && python Training-Generation-Energy.py
```

### Deployment
1. Configure deployment settings in `Deploy.py`:
   ```python
   model_deployment = ManagedOnlineDeployment(
       name="UKEnergyModelDeploy",
       endpoint_name="uk-energy-model-endpoint",
       instance_type="Standard_DS4_v2",
       instance_count=1
   )
   ```

2. Execute deployment:
   ```bash
   jupyter nbconvert --to python "Deploy.ipynb" && python Deploy.py
   ```

### Inference
Sample request to deployed endpoint:
```python
import requests

endpoint = "https://uk-energy-model-endpoint.azureml.net/score"
data = {"input_data": [...]}  # Your time series window

response = requests.post(endpoint, json=data, headers={"Authorization": "Bearer API_KEY"})
print(response.json())
```

## 7. Examples

### Example 1: Training Validation
![Training Validation](/training_validation_loss.png)

### Example 2: Sample Prediction
```python
# Load saved model
model = mlflow.keras.load_model("models/tensorflow_series_UK_energy_v2")

# Make prediction
forecast = model.predict(X_test)
```

### Example 3: Batch Processing
```python
from scripts.inference_score import run

with open("batch_data.json") as f:
    results = run(f.read())
```

## 8. Project Structure

```
Energy-UK/
├── .gitignore
├── requirements.txt            # Full Python dependencies
├── scripts/
│   └── inference_score.py      # Scoring script for deployment
├── Deploy.ipynb                  # Azure ML deployment configuration
├── Training-Energy-Comsumption.ipynb  # LSTM model for energy demand
├── Training-Generation-Energy.ipynb   # GRU model for solar generation
├── uk_electricity.csv      # Grid operational data
├── uk_weather.csv          # Meteorological records
├── training_validation_loss.png   # Image of training validation
└── environment.yml             # Conda environment spec
```

## 9. API Reference

### Inference Endpoint
**POST /score**
- Input: JSON array of time series windows
- Output: Predicted values with confidence intervals

### Model Signatures
**Consumption Model**:
```python
ModelSignature(
    inputs=Schema([
        TensorSpec(np.int16, (-1,), "settlement_period"),
        TensorSpec(np.float16, (-1,), "period_hour"),
        # ... additional features
    ]),
    outputs=Schema([
        TensorSpec(np.float32, (-1,), "nd")
    ])
)
```

## 10. How to Contribute

### Contribution Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open Pull Request

### Coding Standards
- PEP 8 compliance
- Type hints for all functions
- MLflow experiment tracking for all model changes

## 11. Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA errors | Verify CUDA 12.x and cuDNN compatibility |
| Azure auth failures | Run `az login` and check subscription |
| Shape mismatches | Validate window_size parameter matches training |

## 12. Changelog

### [1.0.0] - 2024-06-15
- Initial release with LSTM and GRU models
- Azure ML deployment pipelines
- MLflow experiment tracking

## 13. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 14. Contact
**Project Maintainer**: Carlos Yazid
**Email**: contact@carlospadilla.co  
**Issue Tracking**: [Issues](https://github.com/CarlosYazid/Energy-UK/issues)  

For enterprise support inquiries, please contact our partnerships team.