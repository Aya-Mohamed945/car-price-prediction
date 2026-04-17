import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_clean_data(file_path):
    """Load and clean the car dataset"""
    df = pd.read_csv(file_path)
    
    df.drop_duplicates(inplace=True)
    
    df['Selling_Price'] = np.log(df['Selling_Price'])
    df['Present_Price'] = np.log(df['Present_Price'])
    df['Driven_kms'] = np.log(df['Driven_kms'])
    
    df['Car_Age'] = 2026 - df['Year']
    
    df['Brand'] = df['Car_Name'].str.strip().apply(lambda x: x.split()[0])
    
    df['km_per_year'] = df['Driven_kms'] / (df['Car_Age'] + 1)
    
    df['Is_Manual'] = df['Transmission'].replace({'Automatic': 0, 'Manual': 1})
    df['By_Dealer'] = df['Selling_type'].replace({'Dealer': 1, 'Individual': 0})
    
    df.drop(['Year', 'Selling_type', 'Transmission', 'Car_Name'], axis=1, inplace=True)
    
    return df

def prepare_features(df):
    """Prepare features and target for training"""
    X = df.drop(['Selling_Price'], axis=1)
    y = df['Selling_Price']
    
    return X, y

def split_and_encode(X, y, test_size=0.2, random_state=42):
    """Split data and apply one-hot encoding"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    brand_mean = X_train.groupby('Brand')['Present_Price'].mean()
    X_train['brand_goodwill'] = X_train['Brand'].map(brand_mean)
    X_test['brand_goodwill'] = X_test['Brand'].map(brand_mean)
    X_test['brand_goodwill'].fillna(X_train['brand_goodwill'].mean(), inplace=True)
    
    X_train = X_train.drop(['Brand'], axis=1)
    X_test = X_test.drop(['Brand'], axis=1)
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(X_train[['Fuel_Type']])
    X_test_cat = ohe.transform(X_test[['Fuel_Type']])
    
    num_cols = X_train.drop(['Fuel_Type'], axis=1).to_numpy()
    num_cols_test = X_test.drop(['Fuel_Type'], axis=1).to_numpy()
    
    X_train_encoded = np.concatenate([num_cols, X_train_cat], axis=1)
    X_test_encoded = np.concatenate([num_cols_test, X_test_cat], axis=1)
    
    numeric_feature_names = X_train.drop(['Fuel_Type'], axis=1).columns.tolist()
    fuel_feature_names = ohe.get_feature_names_out(['Fuel_Type']).tolist()
    all_feature_names = numeric_feature_names + fuel_feature_names + ['brand_goodwill']
    
    return X_train_encoded, X_test_encoded, y_train, y_test, ohe, all_feature_names
