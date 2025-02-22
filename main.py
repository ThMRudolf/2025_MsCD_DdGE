"""
    main to prepare data, train a model and validate.
"""
# import modules
from src.prep import (read_data,
                      prepare_train_data,
                      split_train_data,
                      save_prep_data_2_prep,
                      save_col_name)
from src.train import (build_model_estimate_house_pricing,
                       fit_house_pricing_models)

#
TRAIN_LOC_STR = 'data/raw/train.csv'
TEST_LOC_STR = 'data/raw/test.csv'
SUBSAMPLES_LOC_STR = 'data/raw/sample_submission.csv'

# Read all data
try:
    train_df = read_data(TRAIN_LOC_STR)
    test_df = read_data(TEST_LOC_STR)
    sample_df = read_data(SUBSAMPLES_LOC_STR)
except Exception as e_read_raw:
    print(f"Exception occurred during data allocation: {e_read_raw}")

# prepare data
try:
    x, y, numerical_cols, categorical_cols = prepare_train_data(train_df)
except Exception as e:
    print(f"Error occurred during data preparation: {e}")
# split data
try:
    x_train, x_valid, y_train, y_valid=split_train_data(x, y)
except Exception as e:
    print(f"Error occurred during data split: {e}")
# save data in data/prep
# x_train data
try:
    save_prep_data_2_prep(x_train, 'x_train')
except Exception as e:
    print(f"Error occurred during data saving x_train: {e}")
# y_train data
try:
    save_prep_data_2_prep(y_train, 'y_train')
except Exception as e:
    print(f"Error occurred during data saving_ y_train: {e}")
# x_valid data
try:
    save_prep_data_2_prep(x_valid, 'x_valid')
except Exception as e:
    print(f"Error occurred during data saving x_valid: {e}")
# y_valid data
try:
    save_prep_data_2_prep(y_valid, 'y_valid')
except Exception as e:
    print(f"Error occurred during data saving y_valid: {e}")

# y_valid data
try:
    save_col_name(numerical_cols, categorical_cols, 'train')
except Exception as e:
    print(f"Error occurred during data saving y_valid: {e}")

#
# Model build and training
X_TRAIN_LOC_STR = 'data/prep/x_train.csv'
X_VALID_LOC_STR = 'data/prep/x_valid.csv'
Y_TRAIN_LOC_STR = 'data/prep/y_train.csv'
Y_VALID_LOC_STR = 'data/prep/y_valid.csv'
# read data
x_train = read_data(X_TRAIN_LOC_STR)
x_valid = read_data(X_VALID_LOC_STR)
y_train = read_data(Y_TRAIN_LOC_STR)
y_valid = read_data(X_VALID_LOC_STR)
# Read numerical columns
try:
    with open('data/prep/numerical_cols_train.txt', 'r') as file:
        num_cols = file.read().splitlines()
except Exception as e:
    print(f"Error occurred during reading numerical columns: {e}")

try:
    with open('data/prep/categorical_cols_train.txt', 'r') as file:
        cat_cols = file.read().splitlines()
except Exception as e:
    print(f"Error occurred during reading numerical columns: {e}")

try:
    my_pipeline = build_model_estimate_house_pricing(x_train[num_cols], x_train[cat_cols])
except Exception as e:
    print(f"Error occurred during model build: {e}")

try:
    my_pipline_fitted = fit_house_pricing_models(my_pipeline,
                             x_train,
                             y_train
                             )
except Exception as e:
    print(f"Error occurred during model fit: {e}")
