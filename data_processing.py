import pandas as pd
import numpy as np
import os

def load_and_process_data(filepath, year_start, year_end, province_filter=None, municipality_filter=None):
    """
    Loads normalizado.csv, filters for specified elections (Convocatoria),
    and returns a clean, pivoted DataFrame for analysis.
    
    Args:
        filepath: Path to CSV.
        year_start: String name of start Convocatoria (e.g. 'Convocatoria 2015/03')
        year_end: String name of end Convocatoria (e.g. 'Convocatoria 2018/12')
        province_filter: Optional Province name to filter by.
        municipality_filter: Optional Municipality name to filter by.
    """
    print(f"Loading data from {filepath}...")
    # Load with low_memory=False to handle mixed types warning if needed
    df = pd.read_csv(filepath, low_memory=False)

    # CRITICAL FIX: Eliminate Aggregation Rows (Province Totals labeled as 'Dato no disponible')
    if 'Municipio' in df.columns:
        df = df[df['Municipio'] != 'Dato no disponible']
        df = df[df['Municipio'].notna()]

    # Filter Convocatorias
    df_start = df[df['Convocatoria'] == year_start].copy()
    df_end = df[df['Convocatoria'] == year_end].copy()
    
    # Apply Geographic Filters if specified
    if province_filter:
        print(f"Filtering by Province: {province_filter}")
        df_start = df_start[df_start['Provincia'] == province_filter]
        df_end = df_end[df_end['Provincia'] == province_filter]
        
    if municipality_filter:
        print(f"Filtering by Municipality: {municipality_filter}")
        df_start = df_start[df_start['Municipio'] == municipality_filter]
        df_end = df_end[df_end['Municipio'] == municipality_filter]
    
    print(f"Start records ({year_start}): {len(df_start)}")
    print(f"End records ({year_end}): {len(df_end)}")
    
    if len(df_start) == 0 or len(df_end) == 0:
        raise ValueError(f"No records found for selected elections/filters. Check filters.")

    def prepare_year_data(df_year, suffix):
        """Pivots party votes for a specific year."""
        
        # Standardization of party names to reduce columns
        # Map main parties, everything else to 'Otros'
        # Note: New parties might appear in other years, logic tries to be generic but might need updates.
        
        # Helper to map generic names
        def map_party(name):
            if pd.isna(name): return 'Otros'
            # Check for main parties substring
            name_upper = str(name).upper()
            
            # Specific Andalucista/Regional Splitting (User Request: NEVER GROUP)
            
            # 1. Adelante Andalucía
            if 'ADELANTE' in name_upper or 'AA' in name_upper: 
                return 'Adelante Andalucía'
                
            # 2. AxSi
            if 'AXSI' in name_upper or 'ANDALUCIA POR SI' in name_upper:
                return 'AxSi'
            
            # 3. PA
            if 'PA' == name_upper or 'PARTIDO ANDALUCISTA' in name_upper or 'PSA' == name_upper: 
                return 'PA'

            # 3b. CA (Coalición Andalucista - 2008)
            if 'CA' == name_upper or 'COALICION ANDALUCISTA' in name_upper or 'COALICIÓN ANDALUCISTA' in name_upper:
                return 'CA'
                
            # 4. NA (Nación Andaluza)
            if 'NA' == name_upper or 'NACION ANDALUZA' in name_upper or 'NACIÓN ANDALUZA' in name_upper:
                return 'NA'
                
            # 5. Other Andalusians (Preserve Name)
            if 'PCPA' in name_upper: return 'PCPA'
            if 'RISA' in name_upper: return 'RISA'
            if 'IZAR' in name_upper: return 'IZAR'
            if 'PRAO' in name_upper: return 'PRAO'
            if 'PNDEA' in name_upper: return 'PNdeA'
            
            # Main Parties
            if 'PSOE' in name_upper: return 'PSOE'
            if 'PP' in name_upper or 'POPULAR' in name_upper: return 'PP'
            if 'VOX' in name_upper: return 'VOX'
            if 'CIUDADANOS' in name_upper or 'CS' in name_upper: return 'Cs'
            
            if 'PODEMOS' in name_upper: return 'Podemos'
            if 'IULV' in name_upper or 'IZQUIERDA UNIDA' in name_upper or 'IU' in name_upper: return 'IU'
            
            if 'UPYD' in name_upper: return 'Otros' 
            
            return 'Otros'

        # Apply mapping explicitly on 'Partido' (Raw) to preserve granularity (e.g. AxSi vs PA)
        # Previous logic used 'nombre_representativo' which grouped AxSi into PA.
        df_year['ProcessedParty'] = df_year['Partido'].fillna(df_year['nombre_representativo']).apply(map_party)
        
        # Group by Location to aggregate votes per party
        location_cols = ['Provincia', 'Municipio']
        
        pivot = df_year.pivot_table(
            index=location_cols, 
            columns='ProcessedParty', 
            values='Votos', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Group by Location to aggregate votes per party
        location_cols = ['Provincia', 'Municipio']
        
        pivot = df_year.pivot_table(
            index=location_cols, 
            columns='ProcessedParty', 
            values='Votos', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Aggregate Censo and Abstención logic:
        # The data contains multiple rows per Municipality (one per Party, per Mesa/Unit).
        # We assume a 'Unit' is identified by unique combination of stats.
        # This potentially merges identical mesas, but is safer than sum() or first().
        # Ideally, we need a 'Mesa' id.
        
        unique_units = df_year[['Provincia', 'Municipio', 'Censo', 'Abstención', 'Nº votantes']].drop_duplicates()
        
        # Now sum the unique units' stats to get Municipality total
        meta_agg = unique_units.groupby(location_cols)[['Censo', 'Abstención']].sum()
        
        # Add Abstention to pivot
        pivot['Abstencion'] = meta_agg['Abstención']
        pivot['Censo'] = meta_agg['Censo']
        
        # Rename columns
        pivot.columns = [f"{col}_{suffix}" for col in pivot.columns]
        return pivot

    p_start = prepare_year_data(df_start, 'start')
    p_end = prepare_year_data(df_end, 'end')
    
    # Merge on data (Provincia, Municipio)
    combined = p_start.join(p_end, how='inner')
    
    print(f"Combined clean records (Municipalities/Locations): {len(combined)}")
    return combined.fillna(0)

def get_unique_convocatorias(filepath):
    """Helper to get available election dates for Dropdowns."""
    df = pd.read_csv(filepath, usecols=['Convocatoria'], low_memory=False)
    return sorted(df['Convocatoria'].unique())

if __name__ == "__main__":
    df = load_and_process_data("e:/appython/elecciones/andalucia/datos/normalizado.csv", 
                               "Convocatoria 2015/03", 
                               "Convocatoria 2018/12")
    print(df.head())
