import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('data/merged_coffee_data.csv')

# 2. Select numeric columns only
sensory_cols = ['Total Cup Points', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
existing = [c for c in sensory_cols if c in df.columns]

# 3. Create Correlation Matrix
plt.figure(figsize=(10, 8))
corr = df[existing].corr()

# 4. Plot Heatmap
sns.heatmap(corr, annot=True, cmap='YlOrBr', fmt='.2f')
plt.title('How Sensory Attributes Correlate', fontsize=14)

plt.savefig('plots/correlation_heatmap.png')
print("✅ Heatmap saved in plots folder.")
plt.show()