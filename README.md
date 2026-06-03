## Business Problem

Airbnb hosts and internal teams need a reliable way to predict whether a listing will achieve a high booking rate. This project builds a machine learning pipeline that estimates booking probability using listing attributes, host behavior, and pricing signals — enabling hosts to optimize listings and Airbnb to improve search ranking and conversion.

---

## Dataset

- **Training set:** 92,067 Airbnb listings with 61 features
- **Target variable:** Binary — whether a listing achieves a high booking rate
- **Key features:** Room type, host response time, pricing, availability, trust indicators, amenities

---

## Approach

### Feature Engineering
- Engineered 50+ features including `price_per_guest`, `description_len`, `house_rules_len`, `bath_per_bedroom`, trust indicators, and availability windows
- Applied TF-IDF encoding on text fields (name, summary, house_rules)
- Handled missing values via median imputation, zero-filling for fees, and indicator flags for sparse fields
- Encoded categorical features using LabelEncoder and OrdinalEncoder

### Models Evaluated

| Model | Validation AUC |
|---|---|
| Logistic Regression | 0.806 |
| Naive Bayes | 0.770 |
| Decision Tree | 0.806 |
| Random Forest | 0.885 |
| **XGBoost (Final)** | **0.953** |

### Final Model: XGBoost
- Hyperparameter tuning via `RandomizedSearchCV` with 50 iterations and 3-fold cross-validation
- Learning curve analysis from 10,000 to 65,000 listings to assess generalization
- **Final Validation AUC: 0.9526 | Accuracy: 0.9039**

---

## Key Insights

- Hosts who respond within an hour have significantly higher booking rates — fast response time is the strongest behavioral signal
- Entire home listings consistently outperform private and shared rooms
- Listings with verified host identity and profile pictures show higher booking probability
- Holiday demand follows a multi-day window pattern — surges begin 5 to 10 days before and rebound 1 to 3 days after
- Cleaning fees above $100 show a notable drop in booking likelihood

---

## Repository Structure

```
airbnb-booking-prediction/
├── notebooks/
│   └── airbnb_booking_prediction.ipynb   # Full modeling pipeline
├── data/
│   └── predictions.csv                   # Final model predictions
├── reports/
│   └── final_report.pdf                  # Full project report
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/airbnb-booking-prediction.git
cd airbnb-booking-prediction

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebooks/airbnb_booking_prediction.ipynb
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
seaborn
jupyter
```

---

## Results

The final XGBoost model achieved a validation AUC of **0.9526** and accuracy of **0.9039**, outperforming all baseline and intermediate models. Key drivers of high booking probability include host responsiveness, room type, trust verification, and pricing relative to listing capacity.
