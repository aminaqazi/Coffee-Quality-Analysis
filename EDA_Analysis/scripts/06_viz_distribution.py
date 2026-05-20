import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load Data
df = pd.read_csv('data/merged_coffee_data.csv')

# 2. Setup Plot
if not os.path.exists('plots'): os.makedirs('plots')
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 3. Create Histogram
sns.histplot(df['Total Cup Points'], kde=True, color='#6F4E37')

plt.title('Coffee Quality Score Distribution', fontsize=14)
plt.xlabel('Total Cup Points')
plt.ylabel('Number of Samples')

# 4. Save
plt.savefig('plots/distribution_plot.png')
print("✅ Distribution plot saved in plots folder.")
plt.show()