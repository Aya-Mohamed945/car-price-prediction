import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple regression models"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1, max_iter=10000),
        'Ridge': Ridge(alpha=1.0)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        trained_models[name] = model
    
    main_model = trained_models['LinearRegression']
    
    return main_model, scaler, results, trained_models

def save_models(model, scaler, ohe, feature_names, model_dir='models'):
    """Save trained models and preprocessing objects"""
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, f'{model_dir}/car_price_model.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump(ohe, f'{model_dir}/ohe.pkl')
    joblib.dump(feature_names, f'{model_dir}/feature_names.pkl')
    
    print(f"✅ Models saved to '{model_dir}/' directory")

def load_models(model_dir='models'):
    """Load trained models"""
    model = joblib.load(f'{model_dir}/car_price_model.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    ohe = joblib.load(f'{model_dir}/ohe.pkl')
    feature_names = joblib.load(f'{model_dir}/feature_names.pkl')
    
    return model, scaler, ohe, feature_names
