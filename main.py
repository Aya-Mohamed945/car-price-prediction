import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import load_and_clean_data, prepare_features, split_and_encode
from src.model_training import train_models, save_models, load_models
from src.prediction import predict_car_price
from src.utils import print_model_results

def main():
    print("="*60)
    print("CAR PRICE PREDICTION SYSTEM")
    print("="*60)
    
    print("\n?? Loading data...")
    df = load_and_clean_data('data/car data.csv')
    print(f"   Dataset shape: {df.shape}")
    
    print("\n?? Preparing features...")
    X, y = prepare_features(df)
    
    print("\n?? Splitting and encoding...")
    X_train, X_test, y_train, y_test, ohe, feature_names = split_and_encode(X, y)
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    print("\n?? Training models...")
    model, scaler, results, all_models = train_models(X_train, X_test, y_train, y_test)
    
    print_model_results(results)
    
    print("\n?? Saving models...")
    save_models(model, scaler, ohe, feature_names)
    
    print("\n?? Ready for predictions!")
    while True:
        predict_car_price(model, scaler, ohe, feature_names)
        
        another = input("\n?? Predict another car? (y/n): ").lower()
        if another != 'y':
            break
    
    print("\n?? Thank you for using Car Price Prediction System!")
