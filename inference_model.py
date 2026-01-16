import numpy as np
import pandas as pd
from scipy.optimize import minimize

def estimate_transfer_matrix(df_cluster, parties_src, parties_dst, start_suffix, end_suffix):
    """
    Estimates the vote transfer matrix P (dims: n_parties_src x n_parties_dst)
    such that V_dst ~= V_src * P
    
    Arguments:
    df_cluster: DataFrame containing vote COUNTS (or normalized counts).
    parties_src: List of source party names.
    parties_dst: List of destination party names.
    start_suffix: Suffix for source columns (e.g. 'start').
    end_suffix: Suffix for destination columns (e.g. 'end').
    """
    
    # Construct Matrices X (Source) and Y (Target)
    # Rows are Municipalities, Cols are Parties
    X = df_cluster[[f"{p}_{start_suffix}" for p in parties_src]].values
    Y = df_cluster[[f"{p}_{end_suffix}" for p in parties_dst]].values
    
    # Normalize by row sum (Censo) to get percentages [0, 1]
    # This stabilizes the optimization (preventing huge gradients)
    row_sums_X = X.sum(axis=1, keepdims=True)
    row_sums_Y = Y.sum(axis=1, keepdims=True)
    
    # Avoid div by zero
    row_sums_X[row_sums_X == 0] = 1.0
    row_sums_Y[row_sums_Y == 0] = 1.0
    
    X_pct = X / row_sums_X
    Y_pct = Y / row_sums_Y
    
    # Weights based on Censo (Total Votes)
    # Larger municipalities should count more to avoid noise from tiny villages
    # We use the Start Year Censo as weight
    weights = row_sums_X.flatten() 
    # Normalize weights to sum to n_samples for numerical stability in loss
    weights = weights / weights.mean()
    
    n_municipalities, n_src = X_pct.shape
    _, n_dst = Y_pct.shape
    
    # We want to find Matrix P of shape (n_src, n_dst)
    # Flatten P to vector x of size n_src * n_dst
    
    def loss_function(flat_P):
        P = flat_P.reshape(n_src, n_dst)
        Y_pred = X_pct @ P
        
        # Weighted Squared Error
        # Scale residuals by sqrt(weight) so that squaring gives weighted error
        residuals = (Y_pct - Y_pred) * np.sqrt(weights[:, np.newaxis])
        
        return np.sum(residuals**2)
    
    # Constraints
    # 1. Sum of rows of P must be 1 (votes from a source party must go somewhere)
    
    concepts = []
    
    # Iterate over each source party row
    for i in range(n_src):
        def row_sum_constraint(flat_P, row_idx=i):
            P = flat_P.reshape(n_src, n_dst)
            return np.sum(P[row_idx, :]) - 1.0
        
        concepts.append({'type': 'eq', 'fun': row_sum_constraint})
        
    # Bounds: 0 <= p_ij <= 1
    bounds = [(0, 1) for _ in range(n_src * n_dst)]
    
    # Initial guess: Uniform distribution (1/n_dst)
    initial_P = np.full((n_src, n_dst), 1.0 / n_dst).flatten()
    
    # Optimization
    print(f"Optimizing transfer matrix ({n_src}x{n_dst})...")
    result = minimize(loss_function, initial_P, method='SLSQP', bounds=bounds, constraints=concepts, tol=1e-4)
    
    if not result.success:
        print(f"Optimization failed: {result.message}")
    
    P_hat = result.x.reshape(n_src, n_dst)
    
    # Create DataFrame
    transfer_df = pd.DataFrame(P_hat, index=parties_src, columns=parties_dst)
    
    return transfer_df

def run_inference_per_cluster(df, start_suffix='start', end_suffix='end'):
    """
    Runs the estimation for each cluster found in the dataframe.
    Returns a dictionary of Transition Matrices.
    """
    if 'Cluster' not in df.columns:
        raise ValueError("Dataframe must must have 'Cluster' column.")
    
    # Define Party Lists (order matters!)
    cols = df.columns
    def get_parties(suffix):
        return sorted([c.replace(f"_{suffix}", "") for c in cols if f"_{suffix}" in c and 'Censo' not in c and '_pct' not in c])
        
    parties_src = get_parties(start_suffix)
    parties_dst = get_parties(end_suffix)
    
    print(f"Parties Start ({start_suffix}):", parties_src)
    print(f"Parties End ({end_suffix}):", parties_dst)
    
    results = {}
    
    # Run for each Cluster
    unique_clusters = sorted(df['Cluster'].unique())
    
    for cluster_id in unique_clusters:
        # Get label if available
        label = df[df['Cluster'] == cluster_id]['Cluster_Label'].iloc[0] if 'Cluster_Label' in df.columns else f"Cluster {cluster_id}"
        print(f"Processing {label}...")
        
        subset = df[df['Cluster'] == cluster_id]
        
        if len(subset) < len(parties_src): 
            print(f"Warning: Cluster {label} has fewer municipalities ({len(subset)}) than variables. Results may be unstable.")
        
        P_matrix = estimate_transfer_matrix(subset, parties_src, parties_dst, start_suffix, end_suffix)
        results[label] = P_matrix
        
        print(f"Result for {label}:")
        print(P_matrix.round(3))
        print("-" * 30)

    # Run for Global (All Andalucia or filtered scope)
    print("Processing Global...")
    global_P = estimate_transfer_matrix(df, parties_src, parties_dst, start_suffix, end_suffix)
    results['Global'] = global_P
    
    return results

if __name__ == "__main__":
    from data_processing import load_and_process_data
    from clustering import perform_clustering
    
    # Test stub
    df = load_and_process_data("e:/appython/elecciones/andalucia/datos/normalizado.csv", "Convocatoria 2015/03", "Convocatoria 2018/12")
    df = perform_clustering(df, src_suffix='start')
    results = run_inference_per_cluster(df, start_suffix='start', end_suffix='end')
