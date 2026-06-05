# ==========================================
# ALGORITHM 2: DECISION TREE CLASSIFIER (Quality Strategic Thresholds)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def run_decision_tree():
    print("🚀 Running Decision Tree Classifier...")
    
    data_path = 'data/merged_coffee_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Moisture']
    target_points = 'Total Cup Points'
    
    if target_points not in df.columns:
        print(f"❌ Error: Column '{target_points}' not found.")
        return
        
    # Standardized Cleaning Pipeline
    columns_to_clean = features + [target_points]
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[\s%]', '', regex=True).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=columns_to_clean)
    
    if len(df) == 0:
        print("❌ Error: No records left after parsing.")
        return
        
    # Establish Binary Classifications relative to the cohort's Median Quality Score
    median_val = df[target_points].median()
    df['Quality_Tier'] = np.where(df[target_points] >= median_val, 'Premium Quality', 'Standard Quality')
    
    print(f"📊 Ready: Training Decision Tree on {len(df)} records.")
    
    X = df[features]
    y = df['Quality_Tier']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n================ DECISION TREE METRICS ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    
    present_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [le.classes_[idx] for idx in present_classes]
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=target_names))
    
    os.makedirs('plots', exist_ok=True)
    
    # Plot Tree Structure Logic Map
    plt.figure(figsize=(22, 12))
    plot_tree(model, feature_names=features, class_names=le.classes_, filled=True, rounded=True, fontsize=11)
    plt.title("Decision Tree Automated Quality Logic Map", fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/model_decision_tree_flowchart.png', dpi=300)
    plt.show()
    
    # Plot Importance Weights
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    
    sns.barplot(x=importances[indices], y=sorted_features, hue=sorted_features, palette="copper", edgecolor='black', legend=False)
    plt.title("CART Feature Importance Weights: Attributes Driving Class Boundaries", fontsize=11)
    plt.xlabel("Relative Information Gain Weight")
    plt.ylabel("Sensory Attribute Metric")
    
    # 🛠️ Explicit layout adjustment to fix text clipping
    plt.subplots_adjust(left=0.25, bottom=0.12, right=0.95, top=0.90)
    plt.savefig('plots/model_decision_tree_feature_importance.png', dpi=300)
    print("✅ Decision Tree pipeline complete.")
    plt.show()

if __name__ == "__main__":
    run_decision_tree()