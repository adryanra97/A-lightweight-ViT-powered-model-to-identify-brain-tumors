# Brain Tumor Detection AI

**Author: Adryan R A**  
**Year: 2025**

A lightweight Vision Transformer (ViT) powered web application for brain tumor detection in medical images. This application provides a modern, medical-grade interface for uploading brain scan images and receiving real-time AI-powered analysis.

## Features

- **AI-Powered Detection**: Uses Vision Transformer architecture for accurate brain tumor classification
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **Gradio Interface**: Interactive web interface for easy image upload and analysis
- **Real-time Analysis**: Instant results with confidence scores and detailed analysis
- **REST API**: Complete API for integration with other systems
- **Docker Deployment**: Easy containerized deployment with health checks
- **Production Ready**: Includes nginx reverse proxy and production configurations

## Architecture

```
├── src/                    # Core AI model and inference code
│   ├── model.py           # Vision Transformer model implementation
│   └── inference.py       # Production inference module
├── api/                   # Web API application
│   ├── main.py           # FastAPI web server
│   └── gradio_app.py     # Gradio web interface
├── brain_tumor_dataset/  # Training dataset (ignored in git)
├── uploads/              # User uploaded images
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Multi-service deployment
└── deploy.sh           # Deployment automation script
```

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- 2GB disk space

### 1. Clone and Deploy
```bash
# Start the application
./deploy.sh start
```

### 2. Access the Application
- **FastAPI Documentation**: http://localhost:8000/docs
- **Gradio Interface**: Run `python api/gradio_app.py` then visit http://localhost:7860
- **Health Check**: http://localhost:8000/health
- **API Endpoints**: http://localhost:8000

## Deployment Commands

```bash
# Development deployment
./deploy.sh start

# Production deployment (with nginx)
./deploy.sh production

# Check status
./deploy.sh status

# View logs
./deploy.sh logs

# Stop application
./deploy.sh stop

# Restart
./deploy.sh restart

# Clean up resources
./deploy.sh clean
```

## Manual Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
```bash
# Set environment variables
export DEBUG=False
export MODEL_PATH=./models/brain_tumor_vit_model.pth
export UPLOAD_FOLDER=./uploads
export MAX_CONTENT_LENGTH=16777216
```

### Run the Application
```bash
# Start the FastAPI server
cd api
python main.py

# Or use uvicorn for production
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Model Information

### Vision Transformer Architecture
- **Model Type**: Custom Vision Transformer (ViT)
- **Classes**: Binary classification (Tumor/No Tumor)
- **Input Size**: 224x224 RGB images
- **Architecture**: Patch-based transformer with attention mechanisms

### Training Details
- **Dataset**: Brain tumor dataset with labeled images
- **Preprocessing**: Normalization, resizing, data augmentation
- **Training**: Advanced techniques with learning rate scheduling
- **Validation**: Cross-validation with performance metrics

## API Documentation

### Upload and Predict
```bash
POST /predict
Content-Type: multipart/form-data

# Response
{
    "prediction": "tumor" | "no_tumor",
    "confidence": 0.95,
    "analysis": {
        "medical_recommendation": "Further examination recommended",
        "confidence_level": "High",
        "risk_assessment": "Positive findings detected"
    }
}
```

### Batch Prediction
```bash
POST /predict_batch
Content-Type: multipart/form-data

# Upload multiple images for batch processing
```

### Health Check
```bash
GET /health

# Response
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### Model Information
```bash
GET /docs

# Interactive API documentation with all endpoints
```

## Medical Use Disclaimer

**IMPORTANT**: This application is for research and educational purposes only. It should not be used as the sole basis for medical diagnosis or treatment decisions. Always consult with qualified medical professionals for proper diagnosis and treatment.

### Guidelines for Use:
- Use only high-quality brain scan images
- Consider results as supplementary information
- Validate findings with medical experts
- Follow institutional protocols for AI-assisted diagnosis

## Security Features

- **Input Validation**: File type and size restrictions
- **Rate Limiting**: API request throttling
- **Error Handling**: Comprehensive error management
- **Security Headers**: CORS, XSS protection
- **Health Monitoring**: Continuous health checks

## Performance Monitoring

### Metrics Available:
- Prediction accuracy and confidence scores
- Response times and throughput
- Model performance statistics
- System health indicators

### Monitoring Endpoints:
- `/health` - Application health status
- `/docs` - Interactive API documentation
- Container logs via `docker-compose logs`

## Development

### Local Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run in development mode
export DEBUG=True
python api/main.py
```

### Model Training
The main notebook `brain_tumor_detection.ipynb` contains the complete training pipeline:

```bash
# Open the notebook
jupyter notebook brain_tumor_detection.ipynb
```

### Testing
```bash
# Test API endpoints
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

## Directory Structure

```
├── brain_tumor_detection.ipynb   # Main training notebook
├── api/
│   ├── main.py                   # FastAPI web application
│   └── gradio_app.py            # Gradio interface
├── src/
│   ├── model.py                 # ViT model implementation
│   └── inference.py             # Inference engine
├── brain_tumor_dataset/         # Training dataset (ignored)
├── models/                      # Trained model files (ignored)
├── logs/                        # Application logs (ignored)
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Multi-service setup
├── nginx.conf                   # Nginx configuration
├── requirements.txt             # Python dependencies
├── deploy.sh                    # Deployment script
└── README.md                    # This file
```

## Model Performance

Based on the training results in the notebook:
- **Test Accuracy**: 82.35%
- **Best Validation Accuracy**: 92.68%
- **ROC AUC Score**: 0.8839
- **Sensitivity**: 83.87%
- **Specificity**: 80.00%

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the logs: `./deploy.sh logs`
- Verify health: `./deploy.sh status`
- Review documentation above
- Check container status: `docker-compose ps`

## Updates and Maintenance

### Regular Maintenance:
- Monitor application logs
- Update dependencies regularly
- Backup trained models
- Review security settings

### Performance Optimization:
- Monitor response times
- Scale with additional containers if needed
- Optimize model loading for faster startup
- Use GPU acceleration if available

---

**Made for advancing medical AI research**
