import pandas as pd
import plotly.express as px
import os

# 1. Load Data
df = pd.read_csv('data/merged_coffee_data.csv')

# 2. Define a Custom Coffee Color Scale
# From Dark Roast (#3C2A21) to Light Crema (#EAD8C0)
coffee_scale = ["#3C2A21", "#6F4E37", "#A67B5B", "#EAD8C0"]

# 3. Create 3D Plot
fig = px.scatter_3d(
    df, 
    x='Aroma', y='Flavor', z='Acidity',
    color='Total Cup Points',
    color_continuous_scale=coffee_scale,
    title='3D Analysis: Sensory Profiles (Espresso Theme)',
    template='plotly_dark'
)

# 4. Match the PowerBI "Professional" background but keep it Coffee-toned
fig.update_layout(
    paper_bgcolor="#1a1a1a", # Deep charcoal
    plot_bgcolor="#1a1a1a",
    font=dict(color="#EAD8C0") # Cream colored text
)

# 5. Save as PNG
if not os.path.exists('plots'): os.makedirs('plots')
fig.write_image("plots/3d_coffee_espresso_theme.png", width=1200, height=800, scale=2)
print("✅ PNG Saved: 3D Analysis with Coffee Theme")