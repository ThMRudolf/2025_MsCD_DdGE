# train.py
"""
This script calls the train data and builds, fits and tests the model.
The train and fitted model is then stored in /data/inference as house_price_model.pkl
"""
# python modules
import os
import joblib
# own modules
from predict_house_prices import build_model_estimate_house_pricing, fit_house_pricing_models


# Define paths
DATA_PREP_PATH = 'data/prep'
MODEL_OUTPUT_PATH = 'data/inferencia'
MODEL_OUTPUT_PATH = 'house_pricing_model.pkl'

# Ensure the output directory exists
os.makedirs(MODEL_OUTPUT_PATH , exist_ok=True)

# Build and fit the model
model = build_model_estimate_house_pricing(DATA_PREP_PATH)
fitted_model = fit_house_pricing_models(model)

# Save the model to a file
model_filepath = os.path.join(MODEL_OUTPUT_PATH, MODEL_OUTPUT_PATH)
joblib.dump(fitted_model, model_filepath)

print(f"Model saved to {model_filepath}")
