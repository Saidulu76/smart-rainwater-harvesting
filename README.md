# Smart Rainwater Harvesting Management System

**Category:** Water Conservation  
**Difficulty:** Intermediate  
**Time Required:** ~30 hours (team of 2–3)

This repository contains a starter project for the *Smart Rainwater Harvesting Management System*:
- LSTM model (Keras/TensorFlow) to forecast rainfall and tank storage.
- A simple genetic algorithm to compute near-optimal usage schedules that minimize overflow and maximize usage.
- A minimal Streamlit dashboard to visualize current storage, forecasts, and recommendations.

## Structure
```
/data                # dataset (included)
/src                 # training and optimization scripts
/models              # saved model artifacts (created after training)
/app                 # Streamlit app to run dashboard
/notebooks           # example notebook for experiments
requirements.txt
README.md
```

## How to run

1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Train the LSTM model (creates models/lstm_model.h5):
```bash
python src/train_lstm.py --data data/smart_rainwater_data (1).csv --epochs 10
```

3. Run genetic algorithm optimization (outputs a JSON with best schedule):
```bash
python src/ga_optimization.py --data data/smart_rainwater_data (1).csv --model models/lstm_model.h5
```

4. Run dashboard (Streamlit):
```bash
streamlit run app/streamlit_app.py
```

## Notes
- The dataset included is `smart_rainwater_data (1).csv`. Adjust column names in `src/train_lstm.py` if necessary.
- This starter project is intentionally simple and easy to extend:
  - Add better feature engineering, hyperparameter tuning, cross-validation.
  - Replace the GA with a more advanced optimizer or constraints.
  - Connect the Streamlit app to a live feed / sensor API.

## Contact
If you want more features (automated hyperparameter tuning, full CI/CD for GitHub, Dockerfile, or deployment scripts), reply here and I'll add them.

