# Getting Started Guide - Soil Health ML Project

Welcome! This guide walks you through your ML project step-by-step.

## What You Have Now

✅ A complete project structure  
✅ Reusable Python modules for data handling, model training, and evaluation  
✅ Unit tests to verify your code works  
✅ Documentation and configuration  

## Your Next Steps (In Order)

### Step 1: Get a Dataset ⬅️ **YOU ARE HERE**

Before you can train anything, you need data!

#### Option A: Find an Existing Dataset (Recommended)
1. **Kaggle** - Search "soil health" or "crop recommendation"
   - https://www.kaggle.com/datasets
2. **GitHub** - Many researchers share datasets
3. **Government Agricultural Data** - USDA, Ministry of Agriculture websites

#### Option B: Create Sample Data (For testing)
```python
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 200

data = {
    'N': np.random.randint(20, 100, n_samples),
    'P': np.random.randint(5, 50, n_samples),
    'K': np.random.randint(10, 60, n_samples),
    'pH': np.random.uniform(4.5, 8.5, n_samples),
    'Rainfall': np.random.randint(500, 2000, n_samples),
    'Temperature': np.random.uniform(15, 35, n_samples),
    'Humidity': np.random.randint(20, 95, n_samples),
    'Crop_Type': np.random.choice(['Rice', 'Wheat', 'Corn', 'Cotton'], n_samples)
}

df = pd.DataFrame(data)
df.to_csv('data/raw/soil_data.csv', index=False)
print("Sample data created!")
```

### Step 2: Add Your Data to the Project

1. Save your CSV file to: `data/raw/soil_data.csv`
2. Ensure these exact column names:
   - `N`, `P`, `K` (nutrient levels)
   - `pH`, `Rainfall`, `Temperature`, `Humidity` (environmental factors)
   - `Crop_Type` (the crop label)

### Step 3: Run Exploratory Data Analysis (EDA)

```bash
pip install jupyter
jupyter notebook notebooks/01_eda.ipynb
```

### Step 4: Train Your First Model

```bash
cd src
python train.py
```

### Step 5: Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
Vireon-Tech/
├── data/
│   ├── raw/              # Your soil dataset goes here
│   └── processed/        # Preprocessed data
├── notebooks/
│   └── 01_eda.ipynb      # Data analysis
├── src/
│   ├── data.py           # Data loading & preprocessing
│   ├── model.py          # ML model
│   ├── train.py          # Training script
│   └── evaluate.py       # Model evaluation
├── tests/
│   └── test_model.py     # Unit tests
├── models/               # Saved trained models
├── config.yaml           # Configuration
└── requirements.txt      # Dependencies
```

## Quick Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train
cd src
python train.py

# Test
pytest tests/ -v

# Notebook
jupyter notebook notebooks/01_eda.ipynb
```

## Data Format Required

Your CSV file needs 8 columns:

| Column | Type | Example |
|--------|------|---------|
| N | int | 59 |
| P | int | 51 |
| K | int | 49 |
| pH | float | 6.8 |
| Rainfall | int | 1134 |
| Temperature | float | 27.5 |
| Humidity | int | 64 |
| Crop_Type | string | Cotton |

## Next Steps After Training

1. **Feature Engineering**: Add NPK ratios, deficiency indicators
2. **Hyperparameter Tuning**: Improve accuracy
3. **Try Different Algorithms**: XGBoost, SVM, etc.
4. **Build API**: Flask/FastAPI for predictions
5. **Deploy**: Docker, AWS, etc.

## Troubleshooting

- **"No such file: soil_data.csv"** → Create your dataset first
- **"ModuleNotFoundError"** → Run `pip install -r requirements.txt`
- **Low accuracy** → Get more data, improve features

---

**Start with Step 1: Get your dataset ready!** 🚀
