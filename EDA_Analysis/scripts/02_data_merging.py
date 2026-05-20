import pandas as pd
import os

# --- 1. LOAD DATA ---
df_kaggle = pd.read_csv('data/df_1_arabica.csv') 
df_scraped = pd.read_csv('data/arabica_data_scraped.csv')

print(f"Original Kaggle rows: {len(df_kaggle)}")
print(f"Original Scraped rows: {len(df_scraped)}")

# --- 2. ALIGN COLUMNS ---
# Mapping the scraped column names to match the Kaggle structure
df_scraped = df_scraped.rename(columns={
    'Grade': 'Total Cup Points',
    'Country': 'Country of Origin',
    'ICP': 'In-Country Partner',
    'Completed': 'Grading Date'
})

# Add a source tag to identify where the data came from
df_kaggle['Data_Source'] = 'Kaggle'
df_scraped['Data_Source'] = 'Scraped'

# --- 3. MERGE ---
# We stack them. Columns not present in the scraped data will be filled with NaN.
df_merged = pd.concat([df_kaggle, df_scraped], ignore_index=True)

# --- 4. CLEAN IDENTIFIERS ---
# Remove '#' from IDs to ensure they are consistent across both sets
df_merged['ID'] = df_merged['ID'].astype(str).str.replace('#', '', regex=False).str.strip()

# --- 5. SAVE ---
if not os.path.exists('data'): os.makedirs('data')
output_path = 'data/merged_coffee_data.csv'
df_merged.to_csv(output_path, index=False)

print("-" * 30)
print(f"✅ MERGE COMPLETE")
print(f"📊 Total Combined Rows: {len(df_merged)}")
print(f"📁 Saved to: {output_path}")