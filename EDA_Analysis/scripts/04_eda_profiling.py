import pandas as pd

# Load the merged dataset
df = pd.read_csv('data/merged_coffee_data.csv')

print("--- 📊 DATA PROFILE ---")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")

# Check data source distribution
if 'Data_Source' in df.columns:
    print("\nRows per Source:")
    print(df['Data_Source'].value_counts())

# Check for missing values (Top 10 columns)
print("\nMissing Values (Top 10 Columns):")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Basic Info
print("\nColumn Data Types:")
print(df.dtypes)