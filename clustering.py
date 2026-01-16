from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def perform_clustering(df, n_clusters=3, src_suffix='start'):
    """
    Performs K-Means clustering on the STARTING year voting percentages.
    Returns the dataframe with an assigned 'Cluster' column and cluster centers.
    """
    print(f"performing clustering with K={n_clusters}")
    
    # Calculate percentages for Source Year
    cols_src = [c for c in df.columns if f"_{src_suffix}" in c and 'Censo' not in c]
    
    # We must normalize rows so they sum to 1 (proportions of Censo)
    # Denominator is Censo_{src_suffix}
    
    df_pct = df.copy()
    features = []
    
    for col in cols_src:
        # Convert votes to percentage of Censo
        pct_col = f"{col}_pct"
        denominator = df[f'Censo_{src_suffix}']
        # Prevent div by zero
        denominator = denominator.replace(0, 1) # Should not happen in valid data
        
        df_pct[pct_col] = df[col] / denominator
        features.append(pct_col)
    
    # Handle any div by zero or NaNs
    X = df_pct[features].fillna(0)
    
    # TFG likely used the raw percentages. 
    # StandardScaling might be useful but composition data is special. 
    # Let's stick to raw percentages as they are in the same unit [0,1].
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    df['Cluster'] = clusters
    
    # Label Clusters with Descriptive Names
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    
    final_labels = {}
    
    for i in range(n_clusters):
        center = centers.iloc[i]
        
        # Get top 2 features (excluding abstention if it's not dominant) generally
        # to build a string like "PSOE+PP" or "Abstencyion+PSOE"
        
        # Sort features by value
        sorted_feats = center.sort_values(ascending=False)
        top1_name = sorted_feats.index[0].replace(f"_{src_suffix}_pct", "")
        top1_val = sorted_feats.iloc[0]
        
        top2_name = sorted_feats.index[1].replace(f"_{src_suffix}_pct", "")
        top2_val = sorted_feats.iloc[1]
        
        # Build label based on dominance
        if top1_name == 'Abstencion' and top1_val > 0.40:
            label = "Alta Abstención"
        elif top1_val > 0.40:
             label = f"Bastión {top1_name}"
        else:
             # Mixed
             label = f"Mixto {top1_name}-{top2_name}"
             
        final_labels[i] = label
            
    df['Cluster_Label'] = df['Cluster'].map(final_labels)
    
    print("Cluster Centers (Percentages):")
    print(centers)
    print("Assigned Labels:", final_labels)
    
    return df
    
    print("Cluster Centers (Percentages):")
    print(centers)
    print("Assigned Labels:", final_labels)
    
    return df

if __name__ == "__main__":
    # Test stub
    from data_processing import load_and_process_data
    df = load_and_process_data("e:/appython/elecciones/andalucia/datos/normalizado.csv")
    df_clustered = perform_clustering(df)
    print(df_clustered['Cluster_Label'].value_counts())
