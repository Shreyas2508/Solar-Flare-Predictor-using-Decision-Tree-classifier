# Solar Flare Predictor

Predict whether a sunspot region will produce a **C‑class or larger solar flare** in the next 24 hours.

## Problem

Solar flares can disrupt satellites, radio communications, and power grids. Forecasters need reliable, interpretable tools. This project uses a decision tree on public sunspot data to flag flare‑prone regions.

## Dataset

- **Source:** UCI Solar Flare dataset (`flare.data2`)
- **Instances:** 1066 active regions
- **Features:** 10 predictors (Zurich class, spot size, spot distribution, activity, evolution, previous flare activity, historical complexity, area, largest spot area)
- **Target:** Binary – 1 if at least one C‑class, M‑class, or X‑class flare occurs in the next 24h

The data is **highly imbalanced** – only ~17% of days have a C‑class or larger flare.

## Model & Results

| Model | Test Recall (flares) | Note |
|-------|----------------------|------|
| Baseline (default tree) | 0.13 | Misses most flares |
| Tuned (max_depth=1, class_weight='balanced') | **0.815** | Catches 8 out of 10 flares |

The best model is a **decision stump** – a single split on the most informative feature, combined with balanced class weights. Deeper trees overfit to the majority class (no flare) and hurt recall.

**Why a stump works:**  
- Severe class imbalance favours simple rules.  
- A single interpretable split generalises better.  
- `class_weight='balanced'` forces the model to pay attention to rare flares.

## Repository Contents

- `solar_flare_predictor.ipynb` – full notebook (data loading, encoding, training, evaluation)
- `solar_flare_tree.joblib` – saved decision tree model
- `onehot_encoder.joblib` – fitted one‑hot encoder for categorical features
- `requirements.txt` – Python dependencies
- `README.md` – this file

## How to Use

```python
import joblib
import pandas as pd

model = joblib.load('solar_flare_tree.joblib')
encoder = joblib.load('onehot_encoder.joblib')

# New sunspot data (categorical + numeric)
new_data = pd.DataFrame({...})  # must have same columns as original

# Encode categoricals
cat_cols = ['zurich_class', 'spot_size', 'spot_distribution']
encoded = encoder.transform(new_data[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# Combine with numeric features
numeric_cols = ['activity', 'evolution', 'prev_24h_flare_activity',
                'historically_complex', 'became_historically_complex', 'area', 'largest_spot_area']
X_new = pd.concat([new_data[numeric_cols], encoded_df], axis=1)

# Predict
pred = model.predict(X_new)  # 1 = flare expected

```
##Requirements
Python 3.7+
pandas
scikit‑learn
joblib

##Author
Shreyas Shahi



