import pandas as pd
import os

filepath = "e:/appython/elecciones/andalucia/datos/normalizado.csv"
df = pd.read_csv(filepath, low_memory=False)

years = ['Convocatoria 1994/06', 'Convocatoria 2008/03', 'Convocatoria 2015/03']

for year in years:
    print(f"\n--- {year} Party Check ---")
    subset = df[df['Convocatoria'] == year]
    
    # Check 'Partido' column for anything resembling Andalucista
    parties = subset['Partido'].unique()
    matches = [p for p in parties if 'ANDAL' in str(p).upper() or 'PA' in str(p).upper()]
    print("Potential Matches in 'Partido':")
    for m in matches:
        print(f"  - {m}")
        
    # Check 'nombre_representativo'
    reps = subset['nombre_representativo'].unique()
    print("Unique 'nombre_representativo':")
    for r in reps:
        print(f"  - {r}")
