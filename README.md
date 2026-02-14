# Patient Triage Prediction System

A hybrid machine learning system that combines rule-based NEWS2 scoring with ensemble ML models to predict emergency department patient acuity levels.

---

## Overview

This system predicts patient triage acuity levels (Critical, Urgent, Non-Urgent) using a hybrid approach:

- **Rule-based layer**: NEWS2 (National Early Warning Score 2) identifies critical patients
- **ML layer**: Stacking ensemble with Logistic Regression meta-learner for cases not flagged as critical by NEWS2
- **Explainability**: SHAP values for ML predictions and NEWS2 score breakdowns

### Acuity Levels

- **Class 1 (Critical)**: Immediate attention required
- **Class 2 (Urgent)**: Requires prompt medical attention
- **Class 3 (Non-Urgent)**: Can wait for treatment

## Features

**Hybrid Prediction System**

- Rule-based NEWS2 scoring identifies critical cases first
- ML ensemble evaluates remaining cases (may still classify as critical/urgent/non-urgent)

**Explainable AI (XAI)**

- SHAP values for ML predictions
- NEWS2 component breakdowns for rule-based predictions

**Multiple ML Models**

- Random Forest, XGBoost, MLP Neural Network
- Stacking ensemble (Meta Learner: Logistic Regression) for best performance

**Text Processing**

- TF-IDF vectorization of chief complaints

**Interactive Prediction Service**

- User-friendly console interface
- Real-time predictions with explanations

## Project Structure

```
pw2/
├── load_data.py                   # Download dataset from Hugging Face
├── raw_data/                      # Original datasets
│   ├── triage_train.csv
│   ├── triage_valid.csv
│   └── triage_test.csv
│
├── rule_based/                    # Rule-based NEWS2 triage
│   ├── rule_based_triage.py      # NEWS2 scoring logic
│   ├── rule_preprocess_data.py   # Preprocess for NEWS2
│   └── rule_processed_data/      # Processed rule-based data
│
├── ml_model/                      # Machine learning models
│   ├── ml_preprocess.py          # Data preprocessing + TF-IDF
│   ├── train_tune_models.py      # Train/tune individual models
│   ├── stacking_ensemble.py      # Build stacking ensemble
│   ├── ml_processed_data/        # Processed ML data
│   ├── base_models/              # Trained base models
│   └── ensemble_model/           # Final stacking model
│
├── hybrid_triage/                 # Hybrid system
│   ├── hybrid_triage_eval.py     # Evaluate hybrid system
│   ├── hybrid_xai.py             # XAI explanations (SHAP + NEWS2)
│   ├── save_shap_background.py   # Prepare SHAP background data
│   └── xai_outputs/              # SHAP visualizations
│
├── application/                   # Production-ready service
│   └── prediction_service.py     # Interactive prediction service
│
├── analysis/                      # Data analysis scripts
│   ├── analyze_data.py           # Exploratory data analysis
│   ├── analyze_outliers.py       # Outlier detection
│   └── count_dataset.py          # Dataset statistics
│
├── requirements.txt               # Python dependencies (REQUIRED)
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Windows/Linux/MacOS

### 1. Clone the Repository

```bash
git clone <repository-url>
cd pw2
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Important:** The `requirements.txt` file is included in this repository and contains all required dependencies with version specifications.

### 4. Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, catboost, lightgbm, shap; print('All dependencies installed!')"
```

## Execution Flow

### Complete Pipeline (From Scratch)

```
0. Download Dataset (First Time Only)
   └─→ load_data.py
       ├─ Downloads from Hugging Face (dischargesum/triage)
       ├─ Saves to raw_data/triage_train.csv
       ├─           raw_data/triage_valid.csv
       └─           raw_data/triage_test.csv
                    ↓
1. Raw Data (triage_train.csv, triage_valid.csv, triage_test.csv)
                    ↓
2. Rule-based Preprocessing
   └─→ rule_based/rule_preprocess_data.py
       ├─ Clips outliers to valid physiological ranges
       ├─ Saves to rule_processed_data/
       └─ Used for NEWS2 scoring
                    ↓
3. ML Preprocessing
   └─→ ml_model/ml_preprocess.py
       ├─ Cleans chief complaint text (lowercase, remove special chars)
       ├─ Drops ID columns (subject_id, stay_id)
       ├─ Removes duplicate rows
       ├─ Removes rows with missing numeric values
       ├─ Clips outliers to valid physiological ranges
       ├─ Scales numerical features (StandardScaler)
       ├─ Vectorizes chief complaints (TF-IDF: 1000 features)
       ├─ Saves processed data to ml_processed_data/
       └─ Saves tfidf_vectorizer.pkl, scaler.pkl, feature_order.pkl
                    ↓
4. Train/Load Models & Build Ensemble
   └─→ ml_model/stacking_ensemble.py
       ├─ Checks for existing models in base_models/ and ensemble_model/
       ├─ Loads pre-trained base models if found, trains if not
       ├─ Loads pre-trained ensemble if found, trains if not
       ├─ Builds ensemble (Stacking) when training
       ├─ Saves models to base_models/ and ensemble_model/
       └─ Generates ensemble_results_choice_X.csv
                    ↓
5. Prepare SHAP Background (Recommended for XAI)
   └─→ hybrid_triage/save_shap_background.py
       ├─ Samples training data
       ├─ Saves to xai_outputs/shap_background.csv
       └─ Enables SHAP explanations in prediction service (local explanation)
                    ↓
6. Evaluate Hybrid System (Optional - for analysis)
   └─→ hybrid_triage/hybrid_triage_eval.py
       ├─ Combines rule-based + ML predictions
       ├─ Generates SHAP explanations
       ├─ Creates NEWS2 breakdowns
       └─ Saves results to hybrid_triage_results.txt
                    ↓
7. Application Service
   └─→ application/prediction_service.py
       ├─ Loads all trained models
       ├─ Loads SHAP background (if available)
       ├─ Interactive patient input
       └─ Real-time predictions with XAI explanations
```

## How to Run

### Quick Start (Using Pre-trained Models)

If models are already trained:

```bash
cd application
python prediction_service.py
```

### Full Pipeline (Training from Scratch)

#### Step 1: Download Dataset (First Time Only)

```bash
# Install datasets package if not already installed
pip install datasets

# Download the dataset from Hugging Face
python load_data.py
```

_Downloads to `raw_data/` folder. Skip if CSV files already exist._

#### Step 2: Preprocess Data

```bash
# Rule-based preprocessing
cd rule_based
python rule_preprocess_data.py

# ML preprocessing
cd ..\ml_model
python ml_preprocess.py
```

_ML preprocessing takes several minutes for TF-IDF/embedding generation_

#### Step 3: Train Models & Build Ensemble

```bash
# Run stacking ensemble (auto-loads or trains models)
python stacking_ensemble.py
```

**What this does:**

- Checks for existing base models in `base_models/` directory
- Loads pre-trained base models if found
- Trains new base models (MLP, XGBoost, Random Forest) if not found
- Checks for existing ensemble models in `ensemble_model/` directory
- Loads pre-trained ensemble if found
- Trains new ensemble if not found
- Saves all models automatically

**Output:**

- `base_models/` - Trained individual models
- `ensemble_model/` - Ensemble models
- `ensemble_results_choice_X.csv` - Performance comparison

#### Step 4: Prepare SHAP Background (Recommended for XAI)

```bash
cd ..\hybrid_triage
python save_shap_background.py
```

_Required for SHAP explanations in prediction service. Skip for faster setup (NEWS2 explanations only)._

#### Step 5: Evaluate Hybrid System (Optional)

```bash
python hybrid_triage_eval.py
```

_Generates comprehensive evaluation report. Optional - not required for prediction service._

**Output:**

- `hybrid_triage_results.txt` - Performance metrics
- `xai_outputs/` - SHAP visualizations and explanations

#### Step 6: Run Prediction Service

```bash
cd ..\application
python prediction_service.py
```

### Using the Prediction Service

When you run the prediction service, you'll be prompted to enter patient data:

```
======================================================================
PATIENT TRIAGE PREDICTION SERVICE
======================================================================

Please enter patient information:
----------------------------------------------------------------------
Temperature (°F, e.g., 98.6): 102.5
Heart Rate (bpm, e.g., 75): 125
Respiratory Rate (breaths/min, e.g., 16): 28
Oxygen Saturation (%, e.g., 98): 88
Systolic Blood Pressure (mmHg, e.g., 120): 85
Diastolic Blood Pressure (mmHg, e.g., 80): 55
Pain Level (0-10, e.g., 5): 8
Chief Complaint (e.g., 'chest pain, shortness of breath'): severe chest pain difficulty breathing
```

**Output:**

```
======================================================================
TRIAGE PREDICTION RESULT
======================================================================

Prediction: Critical (Class 1)
Confidence: 95.2%
Source: Rule-based (NEWS2)
NEWS2 Score: 12

Explanation:
NEWS2 Score: 12 (High Risk - Immediate Clinical Response)
[Detailed NEWS2 component scores...]
```

## System Components

### 1. Rule-based Layer (NEWS2)

- **Location:** `rule_based/rule_based_triage.py`
- **Purpose:** Identify critical patients using physiological thresholds
- **Scoring:** Respiratory rate, O2 saturation, temperature, blood pressure, heart rate

### 2. ML Layer (Stacking Ensemble)

- **Location:** `ml_model/stacking_ensemble.py`
- **Base Models:** Random Forest, XGBoost, MLP
- **Meta-learner:** Logistic Regression
- **Features:** Vitals + TF-IDF vectorized chief complaints

### 3. Hybrid System

- **Location:** `hybrid_triage/hybrid_triage_eval.py`
- **Logic:**
  - If NEWS2 → Critical (Class 1) → Keep prediction
  - Otherwise → Pass to ML ensemble

### 4. Explainability (XAI)

- **SHAP Values:** Feature importance for ML predictions
- **NEWS2 Breakdown:** Component scores for rule-based predictions
- **Visualizations:** Summary plots, waterfall plots, bar charts

## Model Performance

**Achieved Results (3-class, Test Set):**

**Base Models:**

- Random Forest: 68.54% accuracy
- XGBoost: 69.40% accuracy
- MLP Neural Network: 70.33% accuracy

**Ensemble Model:**

- **Stacking (Logistic Regression): 70.89% accuracy, 67.23% macro F1**



### Essential Files for Production

These files are required to run the prediction service:

**Required (Core Functionality):**

| File                       | Purpose              |
| -------------------------- | -------------------- |
| `stacking_lr_ensemble.pkl` | Final ensemble model |
| `tfidf_vectorizer.pkl`     | Text vectorizer      |
| `scaler.pkl`               | Feature scaler       |
| `feature_order.pkl`        | Column order         |

**Recommended (Full XAI Support):**

| File                  | Purpose           |
| --------------------- | ----------------- |
| `shap_background.csv` | SHAP explanations |

_Without `shap_background.csv`, predictions still work but explanations are limited to NEWS2 scores only._

