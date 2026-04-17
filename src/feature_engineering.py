"""
Feature Engineering Module
Handles creation of derived features from raw data
"""

import pandas as pd
import numpy as np

def create_car_age(df, current_year=2026):
    """
    Calculate car age from manufacturing year
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'Year' column
    current_year : int
        Current year for age calculation
    
    Returns:
    --------
    pandas Series with car ages
    """
    return current_year - df['Year']

def create_km_per_year(df):
    """
    Calculate average kilometers driven per year
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'Driven_kms' and 'Car_Age' columns
    
    Returns:
    --------
    pandas Series with km per year
    """
    return df['Driven_kms'] / (df['Car_Age'] + 1)  # +1 to avoid division by zero

def extract_brand(car_name):
    """
    Extract brand name from car name string
    
    Parameters:
    -----------
    car_name : str
        Full car name like 'Maruti Suzuki Swift'
    
    Returns:
    --------
    str : First word of car name (brand)
    """
    return str(car_name).strip().split()[0]

def create_brand_column(df):
    """
    Create brand column from Car_Name
    
    Parameters:
    -----------
    df : pandas DataFrame
        Must contain 'Car_Name' column
    
    Returns:
    --------
    pandas Series with brand names
    """
    return df['Car_Name'].apply(extract_brand)

def create_binary_encoding(df, column, mapping):
    """
    Create binary encoded column from categorical column
    
    Parameters:
    -----------
    df : pandas DataFrame
        Source dataframe
    column : str
        Column name to encode
    mapping : dict
        Mapping dictionary e.g., {'Manual': 1, 'Automatic': 0}
    
    Returns:
    --------
    pandas Series with encoded values
    """
    return df[column].replace(mapping)

def apply_log_transform(df, columns):
    """
    Apply log transformation to specified columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        Source dataframe
    columns : list
        List of column names to transform
    
    Returns:
    --------
    pandas DataFrame with transformed columns (inplace)
    """
    for col in columns:
        df[col] = np.log(df[col])
    return df

def create_brand_goodwill(X_train, brand_column='Brand', price_column='Present_Price'):
    """
    Calculate brand goodwill based on average present price (no data leakage)
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features
    brand_column : str
        Name of brand column
    price_column : str
        Column to use for calculating goodwill
    
    Returns:
    --------
    dict : Brand to goodwill mapping
    """
    brand_goodwill = X_train.groupby(brand_column)[price_column].mean().to_dict()
    return brand_goodwill

def add_brand_goodwill(X, brand_goodwill_dict):
    """
    Add brand goodwill feature to dataframe
    
    Parameters:
    -----------
    X : pandas DataFrame
        Features dataframe
    brand_goodwill_dict : dict
        Mapping from brand to goodwill value
    
    Returns:
    --------
    pandas DataFrame with added brand_goodwill column
    """
    X = X.copy()
    X['brand_goodwill'] = X['Brand'].map(brand_goodwill_dict)
    return X

def prepare_all_features(df, current_year=2026):
    """
    Complete feature engineering pipeline
    
    Parameters:
    -----------
    df : pandas DataFrame
        Raw dataframe
    current_year : int
        Current year for age calculation
    
    Returns:
    --------
    pandas DataFrame with all engineered features
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Log transformations
    df['Selling_Price'] = np.log(df['Selling_Price'])
    df['Present_Price'] = np.log(df['Present_Price'])
    df['Driven_kms'] = np.log(df['Driven_kms'])
    
    # Car age
    df['Car_Age'] = create_car_age(df, current_year)
    
    # Extract brand
    df['Brand'] = create_brand_column(df)
    
    # km per year
    df['km_per_year'] = create_km_per_year(df)
    
    # Binary encodings
    df['Is_Manual'] = create_binary_encoding(df, 'Transmission', {'Automatic': 0, 'Manual': 1})
    df['By_Dealer'] = create_binary_encoding(df, 'Selling_type', {'Dealer': 1, 'Individual': 0})
    
    # Drop unnecessary columns
    df = df.drop(['Year', 'Selling_type', 'Transmission', 'Car_Name'], axis=1)
    
    return df

# Test the module (runs only when script is executed directly)
if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 40)
    print("Available functions:")
    print("  - create_car_age()")
    print("  - create_km_per_year()")
    print("  - extract_brand()")
    print("  - create_brand_column()")
    print("  - create_binary_encoding()")
    print("  - apply_log_transform()")
    print("  - create_brand_goodwill()")
    print("  - add_brand_goodwill()")
    print("  - prepare_all_features()")
