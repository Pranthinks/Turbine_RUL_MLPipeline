# Turbine RUL MLOps Pipeline

**An end-to-end MLOps solution for predicting Remaining Useful Life (RUL) of turbine engines with comprehensive monitoring, automation, and deployment capabilities.**

## Project Overview

This project implements a **production-ready MLOps pipeline** that predicts the Remaining Useful Life (RUL) of turbine engines using sensor data. The system demonstrates industry-standard practices including automated data pipelines, drift detection, model versioning, comprehensive monitoring, and containerized deployment.

### Key Features

- **Automated MLOps Pipeline**: 6-stage modular architecture with Apache Airflow orchestration
- **Data Drift Detection**: Intelligent pipeline routing based on data quality assessments
- **Experiment Tracking**: MLflow integration for model versioning and hyperparameter management
- **Comprehensive Monitoring**: Grafana + Prometheus stack for real-time pipeline observability
- **Production API**: FastAPI endpoint for real-time predictions with data preprocessing
- **Containerized Deployment**: Full Docker Compose setup with CI/CD integration
- **Intelligent Orchestration**: Conditional pipeline execution (retrain vs. predict based on drift)

## Architecture

### Pipeline Stages
```
1. Data Ingestion      → PostgreSQL data extraction via ETL
2. Drift Detection     → Data quality assessment & pipeline routing
3. Data Transformation → Feature preprocessing & cleaning
4. Feature Engineering → Time-series feature extraction (tsfresh)
5. Model Training      → ML model training with MLflow tracking
6. Model Prediction    → Inference with performance monitoring
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Apache Airflow | Pipeline scheduling & workflow management |
| **Database** | PostgreSQL | Unified data storage from multiple APIs |
| **Experiment Tracking** | MLflow | Model versioning, metrics, and registry |
| **Monitoring** | Grafana + Prometheus | Real-time metrics and dashboards |
| **API** | FastAPI | Model serving and user interface |
| **Containerization** | Docker + Docker Compose | Consistent deployment environment |
| **Feature Engineering** | tsfresh, pandas | Time-series feature extraction |
| **ML Framework** | scikit-learn, joblib | Model training and persistence |

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.8+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/Turbine_RUL_MLPipeline.git
cd Turbine_RUL_MLPipeline
```

### 2. Environment Setup
```bash
# Create environment variables file
cp .env.example .env
# Edit .env with your MLflow credentials
```

### 3. Deploy with Docker
```bash
# Start the complete MLOps stack
docker-compose up -d

# Check service status
docker-compose ps
```

### 4. Access Services
- **API & UI**: http://localhost:8001
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

### 5. Initialize Pipeline
```bash
# Calculate reference data for drift detection (run once)
python run_stage.py 7

# Run complete pipeline
python main.py

# Or run individual stages
python run_stage.py 1  # Data ingestion
python run_stage.py 2  # Drift detection
```

## Pipeline Usage

### Running Individual Stages
```bash
python run_stage.py <stage_number>
```

**Available Stages:**
- `1` - Data Ingestion
- `2` - Drift Detection  
- `3` - Data Transformation
- `4` - Feature Engineering
- `5` - Model Training
- `6` - Model Prediction
- `7` - Calculate Reference Data (setup)

### API Usage
```bash
# Health check
curl http://localhost:8001/health

# Upload test data for prediction
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_test_data.txt"
```

### Expected Data Format
Upload space-separated text files with columns:
```
unit_id time_cycles op_setting_1 op_setting_2 op_setting_3 sensor_1 sensor_2 ... sensor_21
1       1           -0.0007      -0.0004      100.0        518.67   641.82   ...
```

## Monitoring & Observability

### Grafana Dashboards
- **Pipeline Metrics**: Stage execution times, success rates
- **Model Performance**: Prediction accuracy, drift scores  
- **System Health**: Resource utilization, error rates
- **Data Quality**: Feature distributions, missing values

### MLflow Tracking
- Experiment comparison and model versioning
- Hyperparameter optimization history
- Model registry with staging/production promotion
- Artifact storage (models, preprocessors, feature selectors)

### Prometheus Metrics
```python
# Custom metrics examples
pipeline_duration_seconds
model_prediction_accuracy
data_drift_score
api_request_count
```

## ML Pipeline Details

### Drift Detection Logic
```python
if drift_detected:
    # Route to full retraining pipeline
    run_stages([1, 3, 4, 5, 6])  
else:
    # Direct to prediction with current model  
    run_stages([1, 6])
```

### Feature Engineering
- **Time-series features**: Statistical measures, trends, seasonality
- **Rolling window statistics**: Moving averages, standard deviations
- **Domain-specific features**: Sensor correlation, degradation patterns
- **Feature selection**: Automated selection based on importance scores

### Model Training
- **Algorithms**: Random Forest, Gradient Boosting, Neural Networks
- **Hyperparameter tuning**: Grid search with cross-validation
- **Model validation**: Time-based splits for temporal data
- **Ensemble methods**: Voting classifiers for improved robustness

## CI/CD Integration

### Automated Deployment Pipeline
```yaml
# .github/workflows/deploy.yml
trigger: push to main
steps:
  1. Run tests
  2. Build Docker images  
  3. Deploy to staging
  4. Run integration tests
  5. Deploy to production
```

### Testing Strategy
```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# API tests
pytest tests/api/
```

## Performance Metrics

### Model Performance
- **RMSE**: < 15 cycles average prediction error
- **R²**: > 0.85 coefficient of determination
- **MAE**: < 12 cycles mean absolute error

### System Performance  
- **API Latency**: < 200ms average response time
- **Throughput**: 100+ predictions/second
- **Uptime**: 99.9% availability target

## Project Structure

```
Turbine_RUL_MLPipeline/
├── src/Turbine_RUL/           # Core pipeline modules
│   ├── components/            # Stage implementations
│   ├── config/               # Configuration management  
│   ├── entity/               # Data classes
│   └── utils/                # Utility functions
├── monitoring_config/         # Grafana & Prometheus configs
├── static/                   # UI assets
├── temp/                     # Temporary processing files
├── artifacts/                # Model artifacts & outputs
├── fast_api.py              # API server
├── main.py                  # Full pipeline runner
├── run_stage.py             # Individual stage runner
├── docker-compose.yml       # Service orchestration
└── requirements.txt         # Python dependencies
```

## Advanced Features

### Intelligent Pipeline Routing
The system automatically decides between full retraining vs. direct prediction based on:
- Statistical drift detection (KS test, Population Stability Index)
- Feature distribution changes
- Model performance degradation
- Data quality metrics

### Scalable Architecture
- **Horizontal scaling**: Multiple API instances with load balancing  
- **Resource management**: Configurable CPU/memory limits
- **Storage optimization**: Efficient artifact caching and cleanup
- **Network isolation**: Secure Docker networking between services

### Production Considerations
- **Error handling**: Comprehensive exception management and recovery
- **Logging**: Structured logging with correlation IDs  
- **Security**: API rate limiting, input validation, CORS configuration
- **Backup**: Automated model and data backup strategies

## Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/enhancement`)
3. **Commit** changes (`git commit -am 'Add feature'`)
4. **Push** to branch (`git push origin feature/enhancement`)  
5. **Create** Pull Request

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Start development server
uvicorn fast_api:app --reload --port 8001
```

## Prerequisites & Requirements

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 20GB free space
- **CPU**: 4+ cores recommended for optimal performance

### Software Dependencies
- Docker Desktop 4.0+
- Python 3.8+
- Git 2.0+

## Troubleshooting

### Common Issues

**Docker containers won't start:**
```bash
# Check Docker daemon
docker system info

# Clean up containers
docker-compose down --volumes
docker system prune -a
```

**API returns 500 errors:**
```bash
# Check logs
docker-compose logs turbine-api

# Verify artifacts exist
ls -la artifacts/
```

**Grafana dashboards empty:**
```bash
# Restart Prometheus
docker-compose restart prometheus

# Check metrics endpoint
curl http://localhost:9091/metrics
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:
- **Create an Issue**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Join GitHub Discussions for questions and ideas
- **Documentation**: Check the [Wiki](../../wiki) for detailed guides

---

**If this project helped you, please give it a star!**

*Built with care for the MLOps community*