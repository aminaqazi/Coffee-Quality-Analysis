import pandas as pd

# --- STEP 1: DEFINE THE DATA ---
# This is the line that was missing!
try:
    df = pd.read_csv('data/merged_coffee_data.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Could not find 'data/merged_coffee_data.csv'. Check your folder!")

# --- STEP 2: DATA PROFILING ---
print("\n--- 📊 DATA PROFILE ---")
print(f"Total Rows: {len(df)}")

# --- STEP 3: STATISTICAL SUMMARY ---
# We define the columns we want to look at
quality_cols = ['Total Cup Points', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']

# We check which of those columns actually exist in our merged file
existing_quality = [col for col in quality_cols if col in df.columns]

print("\n--- 🔢 STATISTICAL SUMMARY ---")
if existing_quality:
    summary = df[existing_quality].describe()
    print(summary)
else:
    print("⚠️ No quality columns found to analyze.")

# Calculate the 'Spread' (Range) of the scores
if 'Total Cup Points' in df.columns:
    score_range = df['Total Cup Points'].max() - df['Total Cup Points'].min()
    print(f"\nQuality Score Range: {score_range:.2f} points")