# Credit Card Fraud Detection - End-to-End ML Project

🚀 **A complete machine learning pipeline for detecting fraudulent credit card transactions, built with Python, Scikit-learn, and FastAPI.**

## 📊 Project Overview

This project demonstrates an end-to-end machine learning workflow:
- **Data Analysis**: Exploratory data analysis of credit card transactions
- **Model Training**: Random Forest classifier with 99.95% accuracy
- **API Deployment**: Production-ready REST API using FastAPI
- **Real-time Predictions**: Single and batch fraud detection

## 🎯 Key Results

- **Dataset**: 284,807 transactions (492 fraud cases, 284,315 normal)
- **Model Performance**:
  - Accuracy: 99.95%
  - Precision: 96.05% (low false alarms)
  - Recall: 74.49% (catches 3 out of 4 fraud cases)
  - F1-Score: 83.91%

## 🛠️ Tech Stack

- **Python**: Data science and ML development
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning (Random Forest)
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Matplotlib**: Data visualization
- **Jupyter**: Interactive development

## 📁 Project Structure

```
credit-model/
├── main.ipynb              # ML pipeline development
├── fraud_api.py            # FastAPI application
├── test_api.py             # API testing script
├── fraud_detection_model.pkl   # Trained model
├── feature_scaler.pkl      # Feature scaler
├── model_metadata.json     # Model information
├── dataset/
│   └── creditcard.csv      # Training data
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install pandas scikit-learn fastapi uvicorn matplotlib requests joblib
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook main.ipynb
```
Execute all cells to train the model and generate the saved files.

### 3. Start the API
```bash
python fraud_api.py
```
The API will be available at `http://localhost:8000`

### 4. Test the API
```bash
python test_api.py
```

### 5. Interactive Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## 🔍 API Endpoints

### Health Check
```
GET /
```
Returns API status and model information.

### Model Information
```
GET /model-info
```
Returns detailed model metrics and metadata.

### Single Prediction
```
POST /predict
```
Predicts fraud for a single transaction.

**Example Request:**
```json
{
  "Time": 144113.0,
  "V1": -0.5,
  "V2": 0.5,
  "V3": 1.2,
  "V4": -0.8,
  "V5": 0.3,
  "V6": -1.1,
  "V7": 0.9,
  "V8": -0.4,
  "V9": 0.7,
  "V10": -0.6,
  "V11": 1.3,
  "V12": -0.9,
  "V13": 0.2,
  "V14": -1.5,
  "V15": 0.8,
  "V16": -0.3,
  "V17": 1.0,
  "V18": -0.7,
  "V19": 0.4,
  "V20": -1.2,
  "V21": 0.6,
  "V22": -0.1,
  "V23": 0.9,
  "V24": -0.5,
  "V25": 0.2,
  "V26": -0.8,
  "V27": 1.1,
  "V28": -0.4,
  "Amount": 124.00
}
```

**Example Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0123,
  "risk_score": "LOW",
  "confidence": 0.9877,
  "model_version": "1.0.0"
}
```

### Batch Prediction
```
POST /predict-batch
```
Predicts fraud for multiple transactions.

## 📈 Model Insights

### Feature Importance
The most important features for fraud detection:
1. **V14** (17.97%) - Most predictive anonymized feature
2. **V10** (11.55%) - Second most important pattern
3. **V12** (9.63%) - Third key indicator
4. **V4** (9.56%) - Fourth critical feature
5. **V17** (9.51%) - Fifth important pattern

### Key Observations
- **Amount** is surprisingly less important (1.1% importance)
- **V-features** (PCA-transformed) are most predictive
- Fraudsters adapt by using small amounts to avoid detection
- Model focuses on behavioral patterns rather than transaction size

## 🎯 Business Impact

### Fraud Detection Benefits
- **Financial Protection**: Prevent fraudulent transactions
- **Customer Trust**: Reduce false positives (96% precision)
- **Risk Management**: Catch 74% of fraud cases
- **Real-time Processing**: API responds in milliseconds

### Cost-Benefit Analysis
- **High Recall**: Catches majority of fraud attempts
- **High Precision**: Minimizes customer inconvenience
- **Scalable**: API can handle production loads
- **Maintainable**: Clear code and documentation

## 🔧 Technicals

### Data Preprocessing
1. **No missing values** in dataset (rare in real-world data)
2. **Feature scaling** using StandardScaler for consistent ranges
3. **Stratified splitting** to maintain fraud ratio in train/test sets
4. **Imbalanced data handling** using class_weight='balanced'

### Model Selection
- **Random Forest** chosen for:
  - Robust to outliers
  - Handles imbalanced data well
  - Provides feature importance
  - Good baseline performance
  - Less prone to overfitting

### API Design
- **FastAPI** for modern, fast API development
- **Pydantic** for data validation
- **Automatic documentation** with OpenAPI/Swagger
- **Error handling** for production reliability
- **Type hints** for better code quality

## 🚀 Future Enhancements

### Model Improvements
- **Feature Engineering**: Create time-based features (hour, day of week)
- **Advanced Models**: Try XGBoost, Neural Networks
- **Ensemble Methods**: Combine multiple models
- **Online Learning**: Update model with new data

### API Enhancements
- **Authentication**: Secure API access
- **Rate Limiting**: Prevent API abuse
- **Monitoring**: Log predictions and performance
- **Database Integration**: Store predictions and feedback

### Production Considerations
- **Containerization**: Docker deployment
- **Orchestration**: Kubernetes for scaling
- **CI/CD Pipeline**: Automated testing and deployment
- **Model Monitoring**: Track model drift and performance

## 📝 Learning Resources

This project demonstrates concepts from:
- **Machine Learning**: Supervised learning, classification
- **Data Science**: EDA, feature importance, model evaluation
- **Software Engineering**: API design, testing, documentation
- **MLOps**: Model deployment, versioning, monitoring

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

**Built with ❤️ for learning and demonstrating ML engineering skills.**
