# =====================================================================
# MAIN PIPELINE --- COFFEE QUALITY ANALYSIS (PHASE 3)
# =====================================================================

import os
import sys

# Import the model runner functions from your cleanly renamed scripts
try:
    import model_1_linear_regression as lr
    import model_2_decision_tree as dt
    import model_3_naive_bayes as nb
except ImportError as e:
    print(f"❌ Structural Error: Could not locate a model script module. {e}")
    print("Ensure all model scripts are located in the exact same directory as this file.")
    sys.exit(1)

def display_menu():
    print("\n" + "="*65)
    print("☕ COFFEE QUALITY DATA SCIENCE PIPELINE - PHASE 3 RUNNER")
    print("="*65)
    print("1. Run Linear Regression Model (Quality Score Prediction Task)")
    print("2. Run Decision Tree Classifier (Strategic Quality Threshold Rules)")
    print("3. Run Gaussian Naive Bayes Model (Processing Method Fingerprinting)")
    print("4. Execute Complete Project Pipeline (Run All Sequentially)")
    print("5. Exit Framework Environment")
    print("="*65)

def main():
    # Pre-run data asset validation check
    target_data = 'data/merged_coffee_data.csv'
    if not os.path.exists(target_data):
        print(f"❌ Critical Error: '{target_data}' not found.")
        print("Please verify that your clean data file is placed in the data/ folder.")
        sys.exit(1)

    while True:
        display_menu()
        choice = input("Select an option to execute (1-5): ").strip()
        
        if choice == '1':
            lr.run_linear_regression()
        elif choice == '2':
            dt.run_decision_tree()
        elif choice == '3':
            nb.run_naive_bayes()
        elif choice == '4':
            print("\n🔄 Initializing full experimental test run sequence...")
            print("-------------------------------------------------------")
            lr.run_linear_regression()
            dt.run_decision_tree()
            nb.run_naive_bayes()
            print("\n✅ Unified Run Complete! All visualizations saved to the 'plots/' directory.")
        elif choice == '5':
            print("\n👋 Exiting runtime pipeline environment. Good luck with your Phase 3 submission!")
            break
        else:
            print("❌ Invalid input choice. Please enter a valid numerical option (1-5).")

if __name__ == "__main__":
    main()