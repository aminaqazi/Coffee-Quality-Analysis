import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('data/merged_coffee_data.csv')
attributes = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
top_3 = df['Country of Origin'].value_counts().nlargest(3).index
df_avg = df[df['Country of Origin'].isin(top_3)].groupby('Country of Origin')[attributes].mean().reset_index()

# Professional Coffee Palette
colors = ["#D4A373", "#A68A64", "#432818"] 

fig = go.Figure()

for i, country in enumerate(top_3):
    row = df_avg[df_avg['Country of Origin'] == country].iloc[0]
    fig.add_trace(go.Scatterpolar(
        r=row[attributes].values,
        theta=attributes,
        fill='toself',
        name=country,
        line=dict(color=colors[i % len(colors)])
    ))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor="#121212",
    polar=dict(
        bgcolor="#1e1e1e",
        radialaxis=dict(visible=True, range=[7, 9], gridcolor="#444", tickfont=dict(color="#EAD8C0"))
    ),
    title="Coffee Sensory Profiles by Origin",
    font=dict(color="#EAD8C0")
)

fig.write_image("plots/radar_coffee_espresso_theme.png", width=1000, height=800)
print("✅ PNG Saved: Radar Chart with Coffee Theme")