import pandas as pd
import numpy as np
from data_processing import load_and_process_data
from clustering import perform_clustering
from inference_model import run_inference_per_cluster
from visualization import generate_sankey
import os

def main():
    print("Starting TFG Replication Analysis...")
    
    # 1. Load Data
    data_path = "e:/appython/elecciones/andalucia/datos/normalizado.csv"
    df = load_and_process_data(data_path)
    
    # 2. Clustering
    df = perform_clustering(df, n_clusters=3)
    
    # Save clustered data for inspection
    df.to_csv("e:/appython/elecciones/andalucia/resultados_clustering.csv")
    print("Clustered data saved to 'resultados_clustering.csv'")
    
    # 3. Inference
    results = run_inference_per_cluster(df)
    
    # 4. Visualization & Reporting
    report_path = "e:/appython/elecciones/andalucia/analisis_final.md"
    charts_dir = "e:/appython/elecciones/andalucia/graficos"
    os.makedirs(charts_dir, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Análisis de Trasvase de Votos (Andalucía 2015-2018)\n\n")
        f.write("Replica del TFG basada en datos por Municipio (Población).\n\n")
        
        f.write("## 1. Caracterización de Clústeres\n")
        counts = df['Cluster_Label'].value_counts()
        for label, count in counts.items():
            f.write(f"- **{label}**: {count} Municipios\n")
        f.write("\n")
        
        f.write("## 2. Matrices de Transferencia y Visualización\n\n")
        
        for label, matrix in results.items():
            f.write(f"### {label}\n")
            
            # Generate Chart
            chart_filename = f"sankey_{label}.html"
            chart_path = os.path.join(charts_dir, chart_filename)
            generate_sankey(matrix, f"Trasvase de Votos - {label}", chart_path)
            
            f.write(f"[Ver Gráfico Interactivo de Sankey]({chart_path})\n\n")
            
            # Markdown Table
            markdown_table = matrix.round(3).to_markdown()
            f.write(markdown_table)
            f.write("\n\n")
            
            # Key Insights (Automated)
            f.write("#### Observaciones Destacadas:\n")
            
            # High Fidelity (Retention)
            retention = []
            for p in matrix.index:
                if p in matrix.columns:
                    val = matrix.loc[p, p]
                    if val > 0.6:
                        retention.append(f"{p} ({val:.1%})")
            if retention:
                f.write(f"- **Alta Fidelidad**: {', '.join(retention)}\n")
            
            # Fugues to Abstention
            if 'Abstencion' in matrix.columns:
                fugues = []
                for p in matrix.index:
                    if p == 'Abstencion': continue
                    val = matrix.loc[p, 'Abstencion']
                    if val > 0.1: 
                        fugues.append(f"{p} -> Abstención ({val:.1%})")
                if fugues:
                    f.write(f"- **Fugas a la Abstención**: {', '.join(fugues)}\n")
            
            # Transfer to VOX
            if 'VOX' in matrix.columns:
                vox_src = []
                for p in matrix.index:
                    if p == 'VOX': continue
                    val = matrix.loc[p, 'VOX']
                    if val > 0.1:
                         vox_src.append(f"{p} -> VOX ({val:.1%})")
                if vox_src:
                     f.write(f"- **Transferencia a VOX**: {', '.join(vox_src)}\n")
            
            f.write("\n")
            
    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
