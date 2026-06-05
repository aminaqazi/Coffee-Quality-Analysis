# ==========================================
# ALGORITHM 3: GAUSSIAN NAIVE BAYES (Sensory Profiling & Process Classification)
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

def run_naive_bayes():
    print("🚀 Running Naive Bayes Classifier...")
    
    data_path = 'data/merged_coffee_data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Moisture']
    target = 'Processing Method'
    
    if target not in df.columns:
        print(f"❌ Error: Target column '{target}' not found.")
        return
        
    df_nb = df.dropna(subset=[target]).copy()
    
    # Standardized Cleaning Pipeline
    for col in features:
        if col in df_nb.columns:
            df_nb[col] = df_nb[col].astype(str).str.replace(r'[\s%]', '', regex=True).str.strip()
            df_nb[col] = pd.to_numeric(df_nb[col], errors='coerce')
            
    df_nb = df_nb.dropna(subset=features)
    
    # Balance out representation limitations
    class_counts = df_nb[target].value_counts()
    valid_classes = class_counts[class_counts > 1].index
    df_nb = df_nb[df_nb[target].isin(valid_classes)].copy()
    
    if len(df_nb) == 0:
        print("❌ Error: No samples remaining after validation.")
        return
        
    print(f"📊 Ready: Training Naive Bayes on {len(df_nb)} complete records.")
    
    X = df_nb[features]
    y = df_nb[target]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n================ NAIVE BAYES METRICS ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    
    present_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = [le.classes_[idx] for idx in present_classes]
    print(classification_report(y_test, y_pred, labels=present_classes, target_names=target_names, zero_division=0))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(11, 8))
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=target_names, yticklabels=target_names)
    plt.title('Naive Bayes Confusion Matrix: Processing Method Fingerprinting', fontsize=11, fontweight="bold")
    plt.ylabel('True Method')
    plt.xlabel('Predicted Method')
    
    # 🛠️ Explicit layout adjustment to fix text clipping on long process names
    plt.subplots_adjust(bottom=0.25, left=0.25, right=0.95, top=0.90)
    
    plt.savefig('plots/model_naive_bayes_confusion_matrix.png', dpi=300)
    print("✅ Naive Bayes pipeline complete.")
    plt.show()

if __name__ == "__main__":
    run_naive_bayes()