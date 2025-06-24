import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load CSV
df = pd.read_csv('cleaning/SCM-concrete-global.csv')
df.columns = df.columns.str.strip()

# Rename columns to match required schema
df = df.rename(columns={
    'Cement(kg/m3)': 'cement_opc',
    'FA (kg/m3)': 'scm_flyash',
    'GGBFS (kg/m3)': 'scm_ggbs',
    'Fine aggregate(kg/m3)': 'locally_avail_sand',
    'Splitting tensile strength (MPa)': 'tensile_strength',
    'Cylinder compressive strength (MPa)': 'compressive_strength',
    'Elastic modulus (GPa)': 'youngs_modulus',
    'SP (kg/m3)': 'superplasticizer',
    'Water(kg/m3)': 'water',
    'Coarse aggregate(kg/m3)': 'coarse_agg'
})

# Add missing columns with default 0
df['silica_sand'] = 0
df['perc_of_fibre'] = 0
df['aspect_ratio'] = 0
df['elongation'] = 0

# Compute water-binder ratio
binder = df['cement_opc'] + df['scm_flyash'] + df['scm_ggbs']
df['w_b'] = df['water'] / binder

# Compute HRWR/b (superplasticizer per binder)
df['hrwr_b'] = df['superplasticizer'] / binder

# Compute density as sum of all components
df['density'] = df[['cement_opc', 'water', 'coarse_agg', 'locally_avail_sand', 'scm_flyash', 'scm_ggbs']].sum(axis=1)

# Drop rows where essential components are missing
required_columns = [
    'cement_opc', 'scm_flyash', 'scm_ggbs', 'silica_sand', 'locally_avail_sand',
    'w_b', 'hrwr_b', 'perc_of_fibre', 'aspect_ratio', 'tensile_strength',
    'density', 'youngs_modulus', 'elongation', 'compressive_strength'
]

df_cleaned = df.dropna(subset=['cement_opc', 'water', 'compressive_strength'])

# Estimate missing tensile strength
mask = df_cleaned['tensile_strength'].isna()
if mask.any():
    model = LinearRegression()
    train = df_cleaned.dropna(subset=['compressive_strength', 'tensile_strength'])
    if not train.empty:
        model.fit(train[['compressive_strength']], train['tensile_strength'])
        df_cleaned.loc[mask, 'tensile_strength'] = model.predict(df_cleaned.loc[mask, ['compressive_strength']])

# Final cleanup and export
df_final = df_cleaned[required_columns].dropna()
df_final.to_csv('SCM-based-concrete-formated.csv', index=False)

print("✅ Cleaning complete. Saved as 'SCM-based-concrete-formated.csv'")
