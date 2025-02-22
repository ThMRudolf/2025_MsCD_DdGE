# train.py
"""
This script calls the train data and builds, fits and tests the model.
The train and fitted model is then stored in /data/inference as house_price_model.pkl
"""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# builds model
def build_model_estimate_house_pricing(numerical_cols, categorical_cols):
    """
    Builds the preprocessing pipeline and the model.

    Params:
        numerical_cols: List of numerical columns.
        categorical_cols: List of categorical columns.

    Returns:
        pipeline: A pipeline with preprocessing and model.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)
                              ])
    return pipeline

# fits model with train data
def fit_house_pricing_models(pipeline,
                             x_train,
                             y_train
                             ):
    """
    fits the house price depending on different inputs.

    Params:
        pipeline: build model for house price estimation.
        x_train: input data to train/fit the model
        y_train: output data to train/fit the model

    Returns:
        Returns a fitted model (pipeline) to estimated price, 
    """
    pipeline.fit(x_train, y_train)
    return pipeline

# validates model   
def validate_estimation_house_pricing(pipeline, x_valid, y_valid):
    """
    Makes predictions and evaluates the model.

    Params:
        pipeline: The fitted pipeline.
        X_valid: Validation features.
        y_valid: Validation target variable.

    Returns:
        preds: Predictions.
        mae: Mean Absolute Error.
        mape: Mean Absolute Percentage Error.
    """
    preds = pipeline.predict(x_valid)
    mae = mean_absolute_error(y_valid, preds)
    mape = mean_absolute_percentage_error(y_valid, preds)
    return preds, mae, mape
