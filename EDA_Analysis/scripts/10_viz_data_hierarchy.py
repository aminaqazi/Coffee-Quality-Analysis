import pandas as pd
import plotly.express as px

# 1. Load Data
df = pd.read_csv('data/merged_coffee_data.csv')

# 2. Fill NAs for the hierarchy
df['Country of Origin'] = df['Country of Origin'].fillna('Unknown')
df['Processing Method'] = df['Processing Method'].fillna('Not Specified')

# 3. Create Sunburst
fig = px.sunburst(
    df, 
    path=['Data_Source', 'Country of Origin', 'Processing Method'], 
    values='Total Cup Points',
    color='Total Cup Points',
    color_continuous_scale='Blues',
    title='Data Hierarchy: Source > Origin > Method'
)

fig.update_layout(template='plotly_dark')
fig.write_html("plots/data_hierarchy_sunburst.html")
print("✅ Sunburst chart saved.")
fig.show()