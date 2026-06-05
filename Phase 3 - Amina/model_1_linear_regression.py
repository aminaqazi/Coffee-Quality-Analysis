# ==========================================
# ALGORITHM 1: STABILIZED MULTIPLE LINEAR REGRESSION FROM SCRATCH
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------------------------------
# 1. CUSTOM STABILIZED ALGORITHMIC CLASS IMPLEMENTATION
# -------------------------------------------------------------
class MultipleLinearRegressionScratch:
    """
    Custom implementation of Multiple Linear Regression using a stabilized
    Normal Equation (Ridge adjustment) to resolve Singular Matrix / Multicollinearity errors.
    """
    def __init__(self, alpha=1e-4):
        self.weights = None
        self.intercept = None
        self.alpha = alpha  # Tiny stabilization constant to prevent matrix singularity

    def fit(self, X_train, y_train):
        # Insert a leading column of ones to handle the intercept bias term
        X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        
        # Compute structural outer matrix product
        A = X_b.T.dot(X_b)
        
        # Add a tiny ridge value to the diagonal matrix to prevent singular matrix errors
        A_stabilized = A + self.alpha * np.eye(X_b.shape[1])
        
        # Solve the stabilized system via closed-form matrix inversion
        beta = np.linalg.inv(A_stabilized).dot(X_b.T).dot(y_train)
        
        self.intercept = beta[0]
        self.weights = beta[1:]

    def predict(self, X_test):
        # Map linear weights across test instances: y = Xw + c
        return X_test.dot(self.weights) + self.intercept


# -------------------------------------------------------------
# 2. RUN PIPELINE EXECUTION ENGINE
# -------------------------------------------------------------
def run_linear_regression():
    print("🚀 Running Custom Stable Multiple Linear Regression Model from Scratch...")
    
    # Unified feature pool and primary continuous target attribute
    features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Moisture']
    target = 'Total Cup Points'
    
    # Load Data Environment with Automated Fallback Engine
    data_path = 'data/merged_coffee_data.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print("✅ Data file loaded successfully from storage path.")
    else:
        print(f"⚠️ Warning: '{data_path}' not found. Initializing operational synthetic database fallback...")
        np.random.seed(42)
        n_samples = 240
        data = {f: np.random.uniform(7.0, 8.5, n_samples) for f in features[:-1]}
        data['Moisture'] = np.random.uniform(0.0, 0.15, n_samples)
        data['Total Cup Points'] = 12 + 1.3*data['Balance'] + 1.1*data['Acidity'] + 0.9*data['Flavor'] + np.random.normal(0, 0.04, n_samples)
        df = pd.DataFrame(data)
        
    # Standardized Missing and Type-Casting Cleaning Verification
    columns_to_clean = features + [target]
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\s%]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Drop records containing missing indicators across chosen features
    df = df.dropna(subset=columns_to_clean)
    
    if len(df) == 0:
        print("❌ Error: No samples remaining after validation processing checks.")
        return

    # 2. Segment Arrays and Train-Test Split
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Fit Custom Closed-Form Model Architecture
    model = MultipleLinearRegressionScratch(alpha=1e-4)
    model.fit(X_train, y_train)
    
    # 4. Predict Matrix Outputs & Calculate Errors
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    # 5. Extract Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n================ CUSTOM SCRATCH LINEAR REGRESSION RESULTS ================")
    print(f"R² Score (Variance Explained): {r2:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("=========================================================================\n")
    
    # Create target directory for validation plot artifact generation
    os.makedirs('plots', exist_ok=True)
    
    # -------------------------------------------------------------
    # GENERATING ALL 5 VISUALIZATIONS FROM THE NOTEBOOK
    # -------------------------------------------------------------
    
    # --- PLOT 1: ACTUAL VS. PREDICTED FIT ---
    plt.figure(figsize=(7, 5.5))
    plt.scatter(y_test, y_pred, color='#C19A6B', alpha=0.7, edgecolors='#4B3621', s=45, label='Predicted Lots')
    ideal_line = np.linspace(min(y_test) - 0.5, max(y_test) + 0.5, 100)
    plt.plot(ideal_line, ideal_line, color='red', linestyle='--', linewidth=2, label='Perfect Fit (45°)')
    plt.title("Actual vs. Predicted Total Cup Points (Model Fit)", fontsize=11, fontweight="bold", color="#5C4033")
    plt.xlabel("True Total Cup Points Score", fontsize=10)
    plt.ylabel("Predicted Total Cup Points Score", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.gca().set_facecolor("#F7EFE5") 
    plt.tight_layout()  
    plt.savefig('plots/model_1_linear_regression_fit.png', dpi=300)
    plt.close()
    
    # --- PLOT 2: FEATURE COEFFICIENT WEIGHTS ---
    coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.weights}).sort_values(by='Coefficient', ascending=True)
    colors = ["#D8C3A5", "#C19A6B", "#B08968", "#A47148", "#8B5E3C", "#7F5539", "#6F4E37", "#5C4033", "#4B3621", "#3E2723"]
    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors[:len(coefficients)], edgecolor='#3E2723', alpha=0.9)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.title("Linear Regression Coefficients (Feature Weights)", fontsize=11, fontweight="bold", color="#5C4033")
    plt.xlabel("Coefficient Weight Value (Impact Magnitude)", fontsize=10)
    plt.ylabel("Sensory Attribute Metric", fontsize=10)
    plt.grid(axis='x', linestyle=':', alpha=0.3)
    plt.gca().set_facecolor("#F7EFE5")
    for bar in bars:
        width = bar.get_width()
        x_pos = width + 0.01 if width >= 0 else width - 0.05
        plt.text(x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left' if width >= 0 else 'right', fontsize=9, color='#3E2723')
    plt.tight_layout()
    plt.savefig('plots/model_1_linear_regression_coefficients.png', dpi=300)
    plt.close()

    # --- PLOT 3: PREDICTION VALUE TRACKING COMPARISON (FIRST 50 SAMPLES) ---
    plt.figure(figsize=(11, 5))
    plt.plot(y_test[:50], color='black', label='Actual Sample Score', marker='o', linewidth=2)
    plt.plot(y_pred[:50], color='chocolate', label='Scratch Matrix Output', marker='x', linestyle='--')
    plt.title("Sequential Prediction Tracking Evaluation (First 50 Samples)", fontsize=11, fontweight="bold", color="#5C4033")
    plt.xlabel("Test Sub-Sample Instance Index", fontsize=10)
    plt.ylabel("Total Cup Points", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.gca().set_facecolor("#F7EFE5")
    plt.tight_layout()
    plt.savefig('plots/model_1_linear_regression_comparison.png', dpi=300)
    plt.close()
    
    # --- PLOT 4: RESIDUALS SCATTER PLOT ---
    plt.figure(figsize=(7.5, 4.5))
    plt.scatter(y_pred, residuals, alpha=0.6, color='#8B5E3C', edgecolors='#3E2723', s=40)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title("Residual Error Magnitude vs. Predicted Targets", fontsize=11, fontweight="bold", color="#5C4033")
    plt.xlabel("Predicted Total Cup Points", fontsize=10)
    plt.ylabel("Residual Dispersion (Error)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.gca().set_facecolor("#F7EFE5")
    plt.tight_layout()
    plt.savefig('plots/model_1_linear_regression_residuals.png', dpi=300)
    plt.close()
    
    # --- PLOT 5: RESIDUAL DISTRIBUTION HISTOGRAM ---
    plt.figure(figsize=(7.5, 4.5))
    sns.histplot(residuals, kde=True, color='#6F4E37', edgecolor='#3E2723', alpha=0.7)
    plt.title("Distribution of Structural Residual Errors", fontsize=11, fontweight="bold", color="#5C4033")
    plt.xlabel("Calculated Residual Error Value", fontsize=10)
    plt.ylabel("Frequency Population Count", fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.gca().set_facecolor("#F7EFE5")
    plt.tight_layout()
    plt.savefig('plots/model_1_linear_regression_residual_dist.png', dpi=300)
    plt.close()
    
    print("✨ Success! All 5 visualization plots saved safely inside your 'plots/' directory.")

if __name__ == "__main__":
    run_linear_regression()