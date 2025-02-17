# prep.py
"""
    This script prepares data to build a estimator for house priciing 
    and uses the functions defined in predict_house_prices.py
"""

# import general modules
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import my modules
from predict_house_prices import *


# Load the datasets

test_df = pd.read_csv('../data/raw/test.csv')
train_df = pd.read_csv('../data/raw/train.csv')
sample_df = pd.read_csv('../data/raw/sample_submission.csv')

# print information about data shape
print("Full train dataset shape is {}".format(train_df.shape))
print("Full test dataset shape is {}".format(test_df.shape))
print("Full sample sales price dataset shape is {}".format(sample_df.shape))

# Take away (drop) first column (only index)
train_df=train_df.drop('Id', axis= 1)
# and plot info about the data
train_df.info()

# select numeric values
train_df_numeric = select_numerical(train_df)
test_df_numeric = select_numerical(test_df)
# select cathegorical values
train_df_cat = select_categorical(train_df)
test_df_cat = select_categorical(test_df)
# save df to ../data/prep
save_prep_data_2_prep(train_df)
save_prep_data_2_prep(test_df)