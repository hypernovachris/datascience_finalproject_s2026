import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_test, predictions, num_points=100):
    """
    Generates two plots to evaluate model performance.
    num_points limits the line graph so it doesn't become a messy blur.
    """
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Time Series Line Graph ---
    plt.figure(figsize=(14, 5))
    # Plotting a subset of points so the spikes are clearly visible
    plt.plot(y_test.values[:num_points], label='Actual CPU Load', color='blue', linewidth=2)
    plt.plot(predictions[:num_points], label='Predicted CPU Load', color='orange', linestyle='--', linewidth=2)
    
    plt.title('Baseline Model: Actual vs. Predicted CPU Load (First 100 Minutes)')
    plt.xlabel('Time Step (Minutes)')
    plt.ylabel('Scaled CPU Utilization (0 to 1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/baseline_timeseries.png') # Save for your LaTeX report
    plt.show()

    # --- Plot 2: Scatter Plot (Actual vs Predicted) ---
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, predictions, alpha=0.3, color='purple')
    
    # Plot the "Perfect Prediction" diagonal line
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='dashed', label='Perfect Prediction')
    
    plt.title('Overall Accuracy: Actual vs. Predicted')
    plt.xlabel('Actual CPU Utilization')
    plt.ylabel('Predicted CPU Utilization')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reports/baseline_scatter.png') # Save for your LaTeX report
    plt.show()

def create_lag_features(df, feature_col, window_size=5):
    """
    Transforms a time series into a supervised learning dataset.
    Uses the past 'window_size' minutes to predict the next minute.
    """
    # Create an empty dataframe to hold our new lagged features
    df_lagged = pd.DataFrame(index=df.index)
    
    # Create lag columns (t-5, t-4, ..., t-1)
    for i in range(window_size, 0, -1):
        df_lagged[f'lag_{i}'] = df[feature_col].shift(i)
        
    # The target is the current value (t)
    df_lagged['target'] = df[feature_col]
    
    # We must drop NaN values created by the shifting process
    df_lagged = df_lagged.dropna()
    
    # Separate features (X) and target (y)
    X = df_lagged.drop(columns=['target'])
    y = df_lagged['target']
    
    return X, y

def run_baseline(data_dir):
    data_path = Path(data_dir)
    
    print("Loading segmented data...")
    train_df = pd.read_csv(data_path / "train_segments.csv")
    test_df = pd.read_csv(data_path / "test_segments.csv")
    
    # Define how many minutes of history the model gets to see
    WINDOW_SIZE = 5 
    
    # For a robust baseline, we should train/test per segment or across all segments.
    # To keep it standard, we'll pool all train segments together.
    # Note: In a highly rigorous setup, you'd apply the lag function per segment 
    # to avoid creating a lag feature that crosses over two different machines.
    # For this baseline, we group by segment first.
    
    print(f"Creating lag features (Window Size: {WINDOW_SIZE} mins)...")
    
    # Process Train Data
    X_train_list, y_train_list = [], []
    for _, group in train_df.groupby('logical_id'):
        if len(group) > WINDOW_SIZE:
            X, y = create_lag_features(group, 'cpu_scaled', WINDOW_SIZE)
            X_train_list.append(X)
            y_train_list.append(y)
            
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    
    # Process Test Data
    X_test_list, y_test_list = [], []
    for _, group in test_df.groupby('logical_id'):
        if len(group) > WINDOW_SIZE:
            X, y = create_lag_features(group, 'cpu_scaled', WINDOW_SIZE)
            X_test_list.append(X)
            y_test_list.append(y)
            
    X_test = pd.concat(X_test_list)
    y_test = pd.concat(y_test_list)
    
    print("Training Linear Regression Baseline...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print("-" * 30)
    print("BASELINE METRICS")
    print(f"Mean Squared Error (MSE):  {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print("-" * 30)
    
    # Call the plotting function
    print("Generating visualizations...")
    plot_predictions(y_test, predictions)
    
    return model, X_test, y_test, predictions

if __name__ == "__main__":
    run_baseline("data/processed")