import pandas as pd
import numpy as np
import os

# --- 1. SETUP ---
input_file = 'data/arabica_data_scraped.csv'
output_file = 'data/coffee_master_cleaned.csv'

if not os.path.exists(input_file):
    print(f"❌ Error: {input_file} not found.")
else:
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    # --- SAFETY CHECK: PRINT COLUMNS ---
    # This helps us see if the scraper used different names
    print("Found these columns in your file:", df.columns.tolist())

    # --- 2. PRE-CLEANING: STRIP SPACES ---
    # This removes hidden spaces from column headers like "Total Cup Points "
    df.columns = df.columns.str.strip()

    print("Applying 7 Cleaning Operations...")

    # Op 1: Deduplication
    if 'ID' in df.columns:
        df = df.drop_duplicates(subset=['ID'], keep='first')

    # Op 2: ID Sanitization
    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str).str.replace('#', '', regex=False).str.strip()

    # Op 3: Handling Missing Values
    if 'Owner' in df.columns:
        df['Owner'] = df['Owner'].fillna('Not Specified')

    # Op 4: Flexible Column Renaming
    # We check if the column exists before trying to rename it
    rename_dict = {
        'Total Cup Points': 'Total_Score', 
        'Country of Origin': 'Country',
        'Processing Method': 'Processing_Method'
    }
    df.rename(columns=rename_dict, inplace=True)

    # --- CRITICAL FIX FOR YOUR ERROR ---
    # If the rename worked, we use Total_Score. If not, we try the original name.
    score_col = 'Total_Score' if 'Total_Score' in df.columns else 'Total Cup Points'

    if score_col in df.columns:
        # Op 5: Outlier Removal (Tukey's IQR)
        # Ensure the column is numeric first
        df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
        df = df.dropna(subset=[score_col]) # Remove rows where score isn't a number
        
        Q1 = df[score_col].quantile(0.25)
        Q3 = df[score_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        df = df[df[score_col] >= lower_bound]
    else:
        print("⚠️ Warning: Could not find a 'Total Score' column. Skipping Outlier Removal.")

    # Op 6: Date Standardization
    if 'Expiration' in df.columns:
        df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')

    # Op 7: String Cleaning
    for col in ['Country', 'Species']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # --- 4. EXPORT ---
    df.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"✅ CLEANING COMPLETE!")
    print(f"📊 Final Record Count: {len(df)}")
    print(f"📁 Cleaned file saved as: {output_file}")