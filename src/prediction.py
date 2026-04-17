import pandas as pd
import numpy as np

def predict_car_price(model, scaler, ohe, feature_names):
    """Interactive car price prediction"""
    
    print("\n" + "="*50)
    print("Car Price Prediction System")
    print("="*50)
    
    print("\nPlease enter the car details:")
   
    present_price = float(input("   Current price of the car (in thousands): "))
    driven_kms = float(input("   Total kilometers driven: "))
    fuel_type = input("   Fuel type (Petrol/Diesel/CNG): ")
    owner = int(input("   Number of previous owners (0,1,2,3): "))
    car_age = int(input("   Car age in years: "))
    transmission = input("   Transmission type (Manual/Automatic): ")
    selling_type = input("   Selling type (Dealer/Individual): ")
    
    is_manual = 1 if transmission.lower() == 'manual' else 0
    by_dealer = 1 if selling_type.lower() == 'dealer' else 0
    
    km_per_year = driven_kms / (car_age + 1)
    
    print(f"\n📊 (Automatically calculated average km/year: {km_per_year:.0f})")
    
    new_car_df = pd.DataFrame([{
        'Present_Price': present_price,
        'Driven_kms': driven_kms,
        'Fuel_Type': fuel_type,
        'Owner': owner,
        'Car_Age': car_age,
        'km_per_year': km_per_year,
        'Is_Manual': is_manual,
        'By_Dealer': by_dealer
    }])
    
    fuel_encoded = ohe.transform(new_car_df[['Fuel_Type']])
    
    numeric_cols = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age', 
                    'km_per_year', 'Is_Manual', 'By_Dealer']
    numeric_data = new_car_df[numeric_cols].to_numpy()
    
    final_data = np.concatenate([numeric_data, fuel_encoded], axis=1)
    
    print(f"\n📊 Input features: {final_data.shape[1]}")
    print(f"📊 Expected features: {scaler.n_features_in_}")
    
    if final_data.shape[1] < scaler.n_features_in_:
        missing_cols = scaler.n_features_in_ - final_data.shape[1]
        zeros_to_add = np.zeros((final_data.shape[0], missing_cols))
        final_data = np.concatenate([final_data, zeros_to_add], axis=1)
        print(f"✅ Added {missing_cols} zero columns")
    elif final_data.shape[1] > scaler.n_features_in_:
        final_data = final_data[:, :scaler.n_features_in_]
        print(f"✅ Trimmed to {scaler.n_features_in_} features")
    
    final_data_scaled = scaler.transform(final_data)
    predicted_price_log = model.predict(final_data_scaled)[0]
    predicted_price_real = np.exp(predicted_price_log)
    
    print("\n" + "="*50)
    print(f"💰 Predicted selling price: {predicted_price_real:,.2f} thousand")
    print(f"   = {predicted_price_real * 1000:,.0f} EGP")
    print("="*50)
    
    return predicted_price_real

def predict_batch(model, scaler, ohe, cars_data):
    """Predict prices for multiple cars at once"""
    predictions = []
    
    for car in cars_data:
        pass
    
    return predictions
