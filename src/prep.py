# prep.py
"""
    This module prepares data to build a estimator for house pricing 
    It provides several funcitons:
        read_raw_data(): reads the raw data
        select_numerical(): selects numerical columns
        select_categorical(): selects categorical columns
        prepare_train_data():  prepares train data, 
                               uses select_numerical() and select_categorical()
        split_train_data(): splits data into train and validation data (80%/20%)
        save_prep_data_2_prep(): saves data to data/prep folder for training
        
"""

# import general modules
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the datasets
def read_raw_data(location_str, plot_info=False):
    """
    This fcn read a csv file: it can read test data, train data.
    Parameters:
    -----------
    location_str: str
            file location of the csv file
    Returns:
        df:
            data frame with data (e.g. train or test) 
    """
    df = pd.read_csv(location_str)
    df=df.drop('Id', axis= 1)
    print(f'Full dataset shape is {format(df.shape)}')
    if plot_info:
        df.info()
    return df

# selects numerical values of a df
def select_numerical(df):
    """
    Selects only numerical columns from the DataFrame.

    Params:
        df: DataFrame from which numerical columns are to be selected.

    Returns:
        DataFrame containing only numerical columns.
    """
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    return numerical_df

# selects categorical values of a df
def select_categorical(df):

    """
    Selects only categorical columns from the DataFrame.

    Params:
        df: DataFrame from which categorical columns are to be selected.

    Returns:
        DataFrame containing only categorical columns.
    """
    categorical_df = df.select_dtypes(include=['object'])
    return categorical_df

# Separate target variable and features
# drop 'SalePrice' column from input matrix and define it as output vector y
def prepare_train_data(train):
    """
    Prepares the data for training by separating the target variable and features,
    and identifying numerical and categorical columns.

    Params:
        train: Training data.

    Returns:
        X: Features.
        y: Target variable.
        numerical_cols: List of numerical columns.
        categorical_cols: List of categorical columns.
    """
    x = train.drop('SalePrice', axis=1)
    y = train['SalePrice']
    numerical_cols = select_numerical(x).columns
    categorical_cols = select_categorical(x).columns
    return x, y, numerical_cols, categorical_cols

def split_train_data(x, y):
    """
    Estimates the house price depending on different inputs.

    Params:
        train: Train data.
        test: Test data.
        sample: Sales prices of test data.

    Returns:
        Returns a DataFrame that contains the estimated price, 
        the real price of the test data houses, and the absolute
        and percentage differences of real value and estimated values.
    """


    x_train, x_valid, y_train, y_valid = train_test_split(x,
                                                          y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=0)
    return x_train, x_valid, y_train, y_valid
# saves data to data/prep folder

def save_prep_data_2_prep(df, name):
    """
    Takes a DataFrame and tne name. The df is separeted in 
    x and y data and numercial and categorical values.

    Params:
        df: DataFrame from which categorical columns are to be selected.
        name: 'train' or 'test'

    Returns:
        no return values. Saves the DataFrame for train, test and the 
        numerical and categorical information to ../data/prep
    """
    x_df, y_df, numerical_cols, categorical_cols = prepare_train_data(df)
    x_df.to_csv(f'../data/prep/x_{name}.csv', index=False)
    y_df.to_csv(f'../data/prep/y_{name}.csv', index=False)
    # Save numerical columns to a file
    with open(f'../data/prep/numerical_cols_{name}n.txt', encoding="ascii") as f:
        for col in numerical_cols:
            f.write(f"{col}\n")
    with open(f'../data/prep/categorical_cols_{name}.txt', encoding="ascii") as f:
        for col in categorical_cols:
            f.write(f"{col}\n")
