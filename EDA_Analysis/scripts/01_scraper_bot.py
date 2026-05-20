import sys
import time
import random
import pandas as pd
import io
import os
from seleniumbase import Driver

# 1. Setup Folder Structure
if not os.path.exists('data'): 
    os.makedirs('data')

def scrape_cqi_database(target_pages=5):
    print("🚀 Launching Stealth Scraper...")
    
    # uc=True is the 'magic' that bypasses the loading screen/blocks
    driver = Driver(uc=True, headless=False)
    all_dataframes = []

    try:
        # STEP 1: INITIAL LOGIN
        # The script starts here. You have 30 seconds to log in manually.
        driver.get("https://database.coffeeinstitute.org/login")
        print("🔑 ACTION REQUIRED: Please log in to the website in the browser window now...")
        time.sleep(30) 

        for p in range(1, target_pages + 1):
            url = f"https://database.coffeeinstitute.org/coffees/arabica?page={p}"
            print(f"🔗 Page {p}: Navigating to {url}")
            driver.get(url)
            
            # STEP 2: WAIT FOR TABLE
            # We wait up to 20 seconds for the table to actually appear
            try:
                driver.wait_for_element("#DataTables_Table_0", timeout=20)
                time.sleep(random.uniform(5, 8)) # Human-like pause
                
                # OPTIONAL: Try to set entries to 500 if it's the first page
                if p == 1:
                    try:
                        driver.select_option_by_value('select[name="DataTables_Table_0_length"]', "500")
                        print("⚡ Set page to show 500 entries for faster scraping.")
                        time.sleep(5)
                    except:
                        pass

                # STEP 3: EXTRACT DATA
                # We use io.StringIO to prevent the 'No such file' error
                html_source = driver.page_source
                # Targeting the specific table ID from your snippet
                tables = pd.read_html(io.StringIO(html_source), attrs={'id': 'DataTables_Table_0'})
                
                if tables:
                    df = tables[0]
                    # Drop columns that are just icons/empty
                    df = df.dropna(how='all', axis=1)
                    all_dataframes.append(df)
                    print(f"✅ Page {p} SUCCESS: Captured {len(df)} rows.")
                
                # Scroll a bit to look 'human'
                driver.execute_script("window.scrollBy(0, 800);")

            except Exception as table_err:
                print(f"⚠️ Could not find table on page {p}. Website might be slow. Skipping...")
                continue

    finally:
        # STEP 4: CONSOLIDATE & SAVE
        if all_dataframes:
            final_df = pd.concat(all_dataframes, ignore_index=True)
            # Remove any duplicate rows
            final_df = final_df.drop_duplicates()
            
            save_path = "data/arabica_data_scraped.csv"
            final_df.to_csv(save_path, index=False)
            print(f"\n🏆 MISSION COMPLETE!")
            print(f"📊 Total Records Saved: {len(final_df)}")
            print(f"📁 File Location: {save_path}")
        else:
            print("❌ No data was collected. Check your login session.")
            
        driver.quit()

if __name__ == "__main__":
    # If you set entries to 500, you only need 4 pages to get 2000 rows!
    scrape_cqi_database(target_pages=12)