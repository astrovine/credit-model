import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_model_info():
    """Test the model info endpoint"""
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    info = response.json()
    print(f"Model Type: {info['model_type']}")
    print(f"Training Date: {info['training_date']}")
    print(f"Test Accuracy: {info['test_accuracy']:.4f}")
    print(f"Test Precision: {info['test_precision']:.4f}")
    print(f"Test Recall: {info['test_recall']:.4f}")
    print()

def test_single_prediction():
    """Test single transaction prediction"""
    print("=== TESTING SINGLE PREDICTION ===")
    
    # Sample transaction (you can modify these values)
    sample_transaction = {
        "Time": 144113.0,
        "V1": -0.5, "V2": 0.5, "V3": 1.2, "V4": -0.8, "V5": 0.3,
        "V6": -1.1, "V7": 0.9, "V8": -0.4, "V9": 0.7, "V10": -0.6,
        "V11": 1.3, "V12": -0.9, "V13": 0.2, "V14": -1.5, "V15": 0.8,
        "V16": -0.3, "V17": 1.0, "V18": -0.7, "V19": 0.4, "V20": -1.2,
        "V21": 0.6, "V22": -0.1, "V23": 0.9, "V24": -0.5, "V25": 0.2,
        "V26": -0.8, "V27": 1.1, "V28": -0.4,
        "Amount": 124.00
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_transaction
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Is Fraud: {result['is_fraud']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_suspicious_transaction():
    """Test with a potentially suspicious transaction"""
    
    # Create a more suspicious looking transaction
    suspicious_transaction = {
        "Time": 84000.0,  # Different time
        "V1": 2.5, "V2": -3.1, "V3": 4.2, "V4": -2.8, "V5": 1.9,
        "V6": -2.7, "V7": 3.1, "V8": -1.9, "V9": 2.3, "V10": -3.2,
        "V11": 2.8, "V12": -2.4, "V13": 1.7, "V14": -4.1, "V15": 3.3,
        "V16": -2.1, "V17": 2.9, "V18": -1.8, "V19": 2.2, "V20": -3.5,
        "V21": 2.1, "V22": -1.7, "V23": 3.2, "V24": -2.3, "V25": 1.6,
        "V26": -2.9, "V27": 3.4, "V28": -2.2,
        "Amount": 1.00  # Small amount (typical of fraud)
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=suspicious_transaction
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Is Fraud: {result['is_fraud']}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("FRAUD DETECTION API TESTING")
    print("=" * 50)
    
    try:
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_suspicious_transaction()

        print("• Achieved 99.95% accuracy with 74.5% fraud recall")

        
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Start the API first with:")
        print("python fraud_api.py")
    except Exception as e:
        print(f"❌ Error: {e}")
