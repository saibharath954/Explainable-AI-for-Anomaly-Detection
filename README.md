# Fraud Detection System 🔍

A robust, explainable AI system for detecting fraudulent financial transactions using **Isolation Forest**, **SHAP explanations**, and **CTGAN data augmentation**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Performance Metrics](#-performance-metrics)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features
- **Anomaly Detection**: Isolation Forest for unsupervised fraud detection  
- **Explainable AI**: SHAP-based feature attributions for predictions  
- **Data Augmentation**: CTGAN to generate synthetic fraud samples  
- **REST API**: FastAPI backend with endpoints for inference  
- **Interactive Dashboard**: React-based frontend for analysts  
- **Model Monitoring**: Evaluation metrics & visualizations  
- **Production Ready**: Docker containerization and CI/CD integration  

---

## 🏗️ Project Structure
```

xad/
├── data/                  # Data directory
│   ├── raw/               # Original dataset (gitignored)
│   └── processed/          # Processed and split data
├── models/                # Saved models (gitignored)
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01\_EDA.ipynb        # Exploratory Data Analysis
│   └── 02\_modeling\_isolation\_forest.ipynb
├── src/                   # Source code
│   ├── api.py             # FastAPI application
│   ├── config.py          # Configuration settings
│   ├── data\_loader.py     # Data loading utilities
│   ├── detector.py        # Isolation Forest implementation
│   ├── evaluator.py       # Model evaluation metrics
│   ├── explainer.py       # SHAP explanation engine
│   ├── generator.py       # CTGAN data augmentation
│   ├── preprocess.py      # Data preprocessing pipeline
│   └── visualizer.py      # Visualization utilities
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── run\_training.py        # Main training script
└── README.md              # This file

````

---

## 🚀 Quick Start

### 🔧 Prerequisites
- Python 3.9+
- pip
- Git

### 📥 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-system.git
   cd fraud-detection-system

2. **Set up Python environment**

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate environment
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**

   * Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   * Download `creditcard.csv`
   * Place it in `data/raw/creditcard.csv`

---

## 💻 Usage

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 2. Train the Model

```bash
python run_training.py
```

This will:

* Preprocess data
* Train Isolation Forest model
* Generate SHAP explanations
* Train CTGAN for augmentation
* Evaluate model performance
* Save all artifacts

### 3. Start the API Server

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

API available at:

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc → [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔬 Model Details

### Isolation Forest

* **Type**: Unsupervised anomaly detection
* **Params**:

  * `n_estimators`: 200
  * `contamination`: "auto"
  * `random_state`: 42

### SHAP Explanations

* **Method**: SHAP (SHapley Additive exPlanations)
* **Output**: Feature importances, contribution scores, and human-readable explanations

### CTGAN Augmentation

* **Purpose**: Generate synthetic fraud samples (class imbalance handling)
* **Params**: `epochs=300`, `batch_size=500`

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_detector.py

# With coverage
pytest --cov=src
```

---

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 fraud-detection-api
```

---

## 📊 Performance Metrics

Typical performance on Kaggle’s Credit Card Fraud dataset:

* **Precision**: 0.85 – 0.92
* **Recall**: 0.75 – 0.82
* **F1-Score**: 0.80 – 0.87
* **ROC AUC**: 0.95 – 0.98
* **Average Precision**: 0.75 – 0.85

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes

   ```bash
   git commit -m "Add amazing feature"
   ```
4. Push branch

   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

**Dev Setup**

```bash
pip install -r requirements-dev.txt
pre-commit install
```

---

## 🙏 Acknowledgments

* **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Libraries**: Scikit-learn, SHAP, CTGAN, FastAPI
* Inspired by research in anomaly detection & explainable AI

⚠️ *Note: This is a demonstration system for educational purposes. Validate thoroughly before production deployment in financial systems.*
