import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_model_results(results):
    """Print model evaluation results in a nice format"""
    print("\n" + "="*60)
    print("Model Performance Summary")
    print("="*60)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    return results_df

def plot_predictions(y_test, y_pred):
    """Plot actual vs predicted prices"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices (log)')
    plt.ylabel('Predicted Prices (log)')
    plt.title('Actual vs Predicted Car Prices')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plot feature importance for linear model"""
    coefficients = model.coef_
    
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    plt.barh(importance_df['Feature'][:10], importance_df['Coefficient'][:10])
    plt.xlabel('Coefficient Value')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.show()
