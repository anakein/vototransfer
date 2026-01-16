import streamlit as st
import pandas as pd
import os
from data_processing import load_and_process_data, get_unique_convocatorias
from clustering import perform_clustering
from inference_model import run_inference_per_cluster
from visualization import generate_sankey
import plotly.express as px

# Setup
# Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datos", "normalizado.csv")
GRAFICOS_DIR = os.path.join(BASE_DIR, "graficos_web")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

st.set_page_config(page_title="Elecciones Andalucía - Trasvase de Votos", layout="wide")

st.title("🗳️ Análisis de Trasvase de Votos - Andalucía (v2.0 - Partidos Separados)")
st.markdown("""
Esta aplicación permite estimar el flujo de votantes entre dos convocatorias electorales.
Utiliza **K-Means Clustering** y **Regresión Lineal Restringida** para inferir el comportamiento.
""")

# Sidebar
st.sidebar.header("Configuración de Análisis")

# 1. Select Elections
try:
    available_convocatorias = get_unique_convocatorias(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
    
col1, col2 = st.sidebar.columns(2)
start_convocatoria = col1.selectbox("Elección Origen", available_convocatorias, index=available_convocatorias.index("Convocatoria 2015/03") if "Convocatoria 2015/03" in available_convocatorias else 0)
end_convocatoria = col2.selectbox("Elección Destino", available_convocatorias, index=available_convocatorias.index("Convocatoria 2018/12") if "Convocatoria 2018/12" in available_convocatorias else len(available_convocatorias)-1)

# 2. Scope Selection
st.sidebar.subheader("Ámbito Territorial")
scope_mode = st.sidebar.radio("Nivel de Detalle", ["Toda Andalucía", "Por Provincia", "Por Municipio"])

filter_province = None
filter_municipality = None

# Helper to load raw for filters
@st.cache_data
def load_geo_data():
    return pd.read_csv(DATA_PATH, usecols=['Provincia', 'Municipio'], low_memory=False).drop_duplicates()

geo_df = load_geo_data()
provinces = sorted(geo_df['Provincia'].unique().astype(str))

if scope_mode == "Por Provincia":
    filter_province = st.sidebar.selectbox("Selecciona Provincia", provinces)
    
elif scope_mode == "Por Municipio":
    sel_prov = st.sidebar.selectbox("Filtra Provincia (para buscar municipio)", provinces)
    munis = sorted(geo_df[geo_df['Provincia'] == sel_prov]['Municipio'].unique().astype(str))
    filter_municipality = st.sidebar.selectbox("Selecciona Municipio", munis)
    # We also keep filter_province for context if needed, but the main filter is muni
    filter_province = sel_prov

# Execution
if st.button("Ejecutar Análisis", type="primary"):
    with st.spinner("Procesando datos y generando modelo..."):
        try:
            # Logic:
            # If Single Municipality selected: 
            # We CANNOT train the model on just 1 row. 
            # We must train on the PROVINCE data, then apply the coefficients to the MUNICIPALITY.
            
            run_scope_province = filter_province
            run_scope_municipality = None # We run training on broader scope
            
            if scope_mode == "Por Municipio":
                # Train on Province, Visualize Municipality
                run_scope_province = filter_province
                run_scope_municipality = None
                st.info(f"Entrenando modelo con datos de la provincia de {filter_province} para estimar comportamiento en {filter_municipality}...")
            elif scope_mode == "Por Provincia":
                 run_scope_province = filter_province
                 
            # 1. Load Data
            df = load_and_process_data(DATA_PATH, start_convocatoria, end_convocatoria, 
                                     province_filter=run_scope_province, 
                                     municipality_filter=run_scope_municipality)
            
            if len(df) < 5:
                st.warning("Hay muy pocos datos para generar un modelo fiable (menos de 5 municipios).")
            
            # 2. Clustering
            df = perform_clustering(df, n_clusters=3, src_suffix='start')
            
            # Show map or stats
            st.subheader(f"Resultados: {start_convocatoria} -> {end_convocatoria}")
            
            # 3. Inference
            results = run_inference_per_cluster(df, start_suffix='start', end_suffix='end')
            
            # Display
            # Display
            st.markdown("---")
            
            # Helper to calculate absolute votes matrix with totals
            def calculate_absolute_matrix(pct_matrix, df_subset, src_suffix, end_suffix):
                # pct_matrix indices are source parties
                abs_matrix = pct_matrix.copy()
                
                # Get total votes for each source party
                total_votes_src = {}
                for party in pct_matrix.index:
                    col_name = f"{party}_{src_suffix}"
                    if col_name in df_subset.columns:
                        total_votes_src[party] = df_subset[col_name].sum()
                    else:
                        total_votes_src[party] = 0
                
                # Multiply row-wise (Transfer estimates)
                for party in pct_matrix.index:
                    abs_matrix.loc[party] = pct_matrix.loc[party] * total_votes_src[party]
                
                # Rounds to int
                abs_matrix = abs_matrix.round(0).astype(int)
                
                # -- Add Comparisons --
                
                # 1. Comparison Column: Total Votes in Destination Election for this party
                # This replaces the redundant "Total Origen" column
                dest_totals_map = {}
                for party in abs_matrix.index:
                    col_name = f"{party}_{end_suffix}"
                    if col_name in df_subset.columns:
                        val = df_subset[col_name].sum()
                    else:
                        val = 0
                    dest_totals_map[party] = val
                
                # Create the column
                abs_matrix['RESULTADO Final'] = pd.Series(dest_totals_map)

                # 2. Add Bottom Total Row (Sum of estimated transfers) to verify distribution
                sum_row = abs_matrix.drop(columns=['RESULTADO Final']).sum(axis=0)
                sum_row.name = 'SUMA Estimada'
                
                # The 'RESULTADO Final' column sum is not meaningful (sum of all parties results), 
                # but we can leave it or put the total census there.
                # Let's clean it up:
                # The bottom row 'SUMA Estimada' for the 'RESULTADO Final' column could be the Total Valid Votes in End.
                
                # Append row
                abs_matrix = pd.concat([abs_matrix, sum_row.to_frame().T])
                
                # Fill the bottom-right corner with the actual total of Destination votes
                total_dest_votes = sum(dest_totals_map.values())
                abs_matrix.loc['SUMA Estimada', 'RESULTADO Final'] = total_dest_votes
                
                # Ensure all are integers
                abs_matrix = abs_matrix.fillna(0).astype(int)

                # 3. Rename Index to include Source Totals "PP (1,234,567)"
                new_index = []
                for idx in abs_matrix.index:
                    if idx == 'SUMA Estimada':
                        new_index.append(idx)
                    else:
                        val = total_votes_src.get(idx, 0)
                        new_index.append(f"{idx} ({val:,.0f})")
                
                abs_matrix.index = new_index
                
                return abs_matrix

            if scope_mode == "Por Municipio":
                 try:
                    target_row = df.loc[(filter_province, filter_municipality)]
                    target_label = target_row['Cluster_Label']
                    
                    st.success(f"📍 **{filter_municipality}** se clasifica en el grupo: **{target_label}**")
                    
                    matrix_pct = results[target_label]
                    
                    # Chart
                    chart_path = os.path.join(GRAFICOS_DIR, "temp_sankey.html")
                    generate_sankey(matrix_pct, f"Estimación para {filter_municipality}", chart_path, 
                                    start_label=start_convocatoria, end_label=end_convocatoria)
                    
                    with open(chart_path, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=600)
                    
                    # Table Pct
                    st.subheader("📊 Porcentajes de Transferencia")
                    st.caption("Filas: Partido en Origen. Columnas: Partido en Destino. (Ej: % del voto de filas que fue a columnas)")
                    st.dataframe(matrix_pct.style.background_gradient(axis=None, cmap='Oranges').format("{:.1%}"), use_container_width=True)
                    
                    # Table Abs
                    subset_muni = df.loc[[ (filter_province, filter_municipality) ]]
                    matrix_abs = calculate_absolute_matrix(matrix_pct, subset_muni, 'start', 'end')
                    
                    st.subheader("🗳️ Estimación de Votos (Absolutos)")
                    st.markdown("""
                    > **¿Cómo leer esta tabla?**
                    > *   **Filas**: ¿Qué votaron antes? (Elección de Origen)
                    > *   **Columnas**: ¿Qué han votado ahora? (Elección de Destino)
                    > *   **Número**: Votantes estimados que han hecho ese cambio.
                    """)
                    st.dataframe(matrix_abs.style.format("{:,}"), use_container_width=True) # Thousands separator
                        
                 except KeyError:
                    st.error("No se encontraron datos para el municipio seleccionado.")
            
            else:
                # Multi-view
                labels = ['Global'] + sorted([l for l in results.keys() if l != 'Global'])
                
                tabs = st.tabs([str(l) for l in labels])
                
                for i, label in enumerate(labels):
                    with tabs[i]:
                        st.header(f"Análisis: {label}")
                        
                        subset = df if label == 'Global' else df[df['Cluster_Label'] == label]
                        
                        if label != 'Global':
                            count = len(subset)
                            st.info(f"Este grupo incluye {count} municipios con comportamiento similar.")
                        
                        matrix_pct = results[label]
                        
                        # Sanitize filename (remove special chars)
                        safe_label = "".join([c for c in label if c.isalnum() or c in (' ', '_', '-')]).strip()
                        path = os.path.join(GRAFICOS_DIR, f"{safe_label}.html")
                        
                        generate_sankey(matrix_pct, f"{label}", path, start_label=start_convocatoria, end_label=end_convocatoria)
                        
                        with open(path, 'r', encoding='utf-8') as f:
                            st.components.v1.html(f.read(), height=600)
                        
                        st.subheader("📊 Porcentajes")
                        st.caption(f"De {start_convocatoria} (Filas) a {end_convocatoria} (Columnas)")
                        st.dataframe(matrix_pct.style.background_gradient(axis=None, cmap='Blues').format("{:.1%}"), use_container_width=True)
                        
                        st.subheader("🗳️ Votos Estimados (Detalle)")
                        st.info("""
                        **Guía de lectura:**
                        - **Filas (Izquierda)**: Partido al que votaron en la **Primera Elección** (Origen).
                        - **Columnas (Arriba)**: Partido al que votaron en la **Segunda Elección** (Destino).
                        - **Celda**: Número de personas que cambiaron su voto.
                        
                        *Ejemplo: La cifra en la fila 'PSOE' y columna 'VOX' son los antiguos votantes del PSOE que ahora votan a VOX.*
                        """)
                        matrix_abs = calculate_absolute_matrix(matrix_pct, subset, 'start', 'end')
                        st.dataframe(matrix_abs.style.format("{:,}"), use_container_width=True)

            st.markdown("---")
            
            if 'Cluster_Label' in df.columns:
                st.write("### Distribución de clústeres")
                st.bar_chart(df['Cluster_Label'].value_counts())

        except Exception as e:
            st.error(f"Error durante el análisis: {e}")
            # st.exception(e) # Uncomment for debug

